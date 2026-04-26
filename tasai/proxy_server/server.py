"""
TAS-AI NICE Proxy Server

A FastAPI server that mediates between TAS-AI and NICE instrument control.
Can operate in multiple modes:
- live: Forward commands to real NICE
- simulation: Use internal TAS simulator
- replay: Replay recorded experiment data
- passthrough: Forward with logging (for debugging)

This separation provides:
1. Safety layer between AI and instrument
2. Ability to test without real instrument
3. Recording and replay of experiments
4. Remote operation capability
"""

import asyncio
import logging
import time
from typing import Optional, Dict, List
from enum import Enum
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "FastAPI components are unavailable. Install TAS-AI with the 'server' extra:\n"
        "  pip install 'tasai[server]'\n"
        "or add fastapi>=0.104 and pydantic>=2 to your environment."
    ) from exc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data models
# =============================================================================

class ProxyMode(str, Enum):
    LIVE = "live"
    SIMULATION = "simulation"
    REPLAY = "replay"
    PASSTHROUGH = "passthrough"


class ProxyConfig(BaseModel):
    mode: ProxyMode = ProxyMode.SIMULATION
    nice_host: Optional[str] = None
    nice_port: Optional[int] = None
    simulation_function: str = "ferromagnetic_magnon"
    simulation_params: Dict = {}
    replay_file: Optional[str] = None


class PositionResponse(BaseModel):
    h: float
    k: float
    l: float
    E: float
    angles: Optional[Dict[str, float]] = None


class AnglesRequest(BaseModel):
    h: float
    k: float
    l: float
    E: float


class ValidateRequest(BaseModel):
    h: float
    k: float
    l: float
    E: float


class ValidateResponse(BaseModel):
    valid: bool
    reason: str = ""


class MeasureRequest(BaseModel):
    h: float
    k: float
    l: float
    E: float
    angles: Optional[Dict[str, float]] = None
    count_time: float = 60.0
    monitor: Optional[int] = None


class MeasureResponse(BaseModel):
    counts: int
    monitor_counts: Optional[int]
    intensity: float
    uncertainty: float
    timestamp: float
    metadata: Dict = {}


class TrajectoryPoint(BaseModel):
    h: float
    k: float
    l: float
    E: float
    angles: Optional[Dict[str, float]] = None
    count_time: float = 60.0


class Trajectory(BaseModel):
    type: str
    points: List[TrajectoryPoint]


class DryRunResponse(BaseModel):
    valid: bool
    warnings: List[str]
    estimated_time: float
    num_points: int


# =============================================================================
# Global state
# =============================================================================

config = ProxyConfig()
simulator = None
geometry = None
nice_connection = None
replay_data = None
replay_index = 0


# =============================================================================
# FastAPI app
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup on startup/shutdown."""
    logger.info("Starting TAS-AI NICE Proxy Server")
    initialize_simulator()
    yield
    logger.info("Shutting down proxy server")


app = FastAPI(
    title="TAS-AI NICE Proxy",
    description="Proxy server for autonomous TAS experiments",
    version="0.1.0",
    lifespan=lifespan
)


def initialize_simulator():
    """Initialize the internal simulator."""
    global simulator, geometry
    
    # Import here to avoid circular imports
    from tasai.instrument.base import TASGeometry
    from tasai.instrument.simulator import (
        TASSimulator, ferromagnetic_magnon, antiferromagnetic_2d
    )
    
    # Default geometry (can be reconfigured)
    geometry = TASGeometry(
        lattice_params=(3.9, 3.9, 13.0, 90, 90, 90),
        orientation=((1, 0, 0), (0, 1, 0)),
        ei_fixed=True,
        fixed_energy=14.7
    )
    
    # Select S(Q,ω) function based on config
    sqw_functions = {
        'ferromagnetic_magnon': ferromagnetic_magnon,
        'antiferromagnetic_2d': antiferromagnetic_2d,
    }
    
    sqw_func = sqw_functions.get(
        config.simulation_function, 
        ferromagnetic_magnon
    )
    
    # Apply any custom parameters
    if config.simulation_params:
        original_func = sqw_func
        params = config.simulation_params
        sqw_func = lambda h, k, l, E: original_func(h, k, l, E, **params)
    
    simulator = TASSimulator(
        geometry=geometry,
        sqw_function=sqw_func,
        background=0.1,
        intensity_scale=1000.0,
        time_scale=100.0  # 100x speedup for testing
    )
    
    logger.info(f"Simulator initialized with {config.simulation_function}")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/status")
async def get_status():
    """Get proxy server status."""
    return {
        "status": "running",
        "mode": config.mode.value,
        "nice_connected": nice_connection is not None,
        "simulator_ready": simulator is not None,
        "total_measurements": len(simulator.measurement_history) if simulator else 0
    }


@app.get("/position", response_model=PositionResponse)
async def get_position():
    """Get current instrument position."""
    if config.mode == ProxyMode.SIMULATION:
        pos = simulator.get_current_position()
        return PositionResponse(
            h=pos.h, k=pos.k, l=pos.l, E=pos.E,
            angles=pos.angles
        )
    
    elif config.mode == ProxyMode.LIVE:
        if not nice_connection:
            raise HTTPException(status_code=503, detail="NICE not connected")
        # Query NICE for position
        position = await query_nice_position()
        return position
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Mode {config.mode} not supported for position query"
        )


@app.post("/calculate_angles")
async def calculate_angles(request: AnglesRequest) -> Dict[str, float]:
    """Calculate instrument angles for (h,k,l,E)."""
    if geometry is None:
        raise HTTPException(status_code=500, detail="Geometry not initialized")
    
    try:
        angles = geometry.hkl_to_angles(request.h, request.k, request.l, request.E)
        return angles
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validate", response_model=ValidateResponse)
async def validate_point(request: ValidateRequest):
    """Validate if measurement point is achievable."""
    if geometry is None:
        raise HTTPException(status_code=500, detail="Geometry not initialized")
    
    accessible = geometry.is_accessible(request.h, request.k, request.l, request.E)
    
    if accessible:
        return ValidateResponse(valid=True)
    else:
        return ValidateResponse(valid=False, reason="Point not accessible")


@app.post("/measure", response_model=MeasureResponse)
async def measure(request: MeasureRequest):
    """Execute a measurement."""
    logger.info(f"Measure request: h={request.h:.3f}, k={request.k:.3f}, "
                f"l={request.l:.3f}, E={request.E:.2f}, t={request.count_time}s")
    
    if config.mode == ProxyMode.SIMULATION:
        # Use internal simulator
        from tasai.instrument.base import MeasurementPoint
        
        point = MeasurementPoint(
            h=request.h,
            k=request.k,
            l=request.l,
            E=request.E,
            angles=request.angles,
            count_time=request.count_time,
            monitor=request.monitor
        )
        
        try:
            result = simulator.measure(point)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        return MeasureResponse(
            counts=result.point.counts,
            monitor_counts=result.point.monitor_counts,
            intensity=result.intensity,
            uncertainty=result.uncertainty,
            timestamp=result.point.timestamp,
            metadata=result.metadata
        )
    
    elif config.mode == ProxyMode.LIVE:
        if not nice_connection:
            raise HTTPException(status_code=503, detail="NICE not connected")
        result = await send_nice_measurement(request)
        return result
    
    elif config.mode == ProxyMode.REPLAY:
        return get_next_replay_result()
    
    elif config.mode == ProxyMode.PASSTHROUGH:
        logger.info(f"PASSTHROUGH: {request}")
        if not nice_connection:
            raise HTTPException(status_code=503, detail="NICE not connected")
        result = await send_nice_measurement(request)
        logger.info(f"PASSTHROUGH result: {result}")
        return result
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {config.mode}")


@app.post("/trajectory/dryrun", response_model=DryRunResponse)
async def dryrun_trajectory(trajectory: Trajectory):
    """Dry-run a trajectory to validate and estimate timing."""
    total_time = 0.0
    warnings = []
    
    prev_angles = None
    
    for i, point in enumerate(trajectory.points):
        # Validate accessibility
        accessible = geometry.is_accessible(point.h, point.k, point.l, point.E)
        if not accessible:
            warnings.append(
                f"Point {i} not accessible: ({point.h}, {point.k}, {point.l}, {point.E})"
            )
            continue
        
        # Calculate angles if not provided
        if point.angles is None:
            try:
                point.angles = geometry.hkl_to_angles(
                    point.h, point.k, point.l, point.E
                )
            except ValueError as e:
                warnings.append(f"Point {i}: {e}")
                continue
        
        # Estimate move time
        if prev_angles:
            move_time = estimate_move_time(prev_angles, point.angles)
        else:
            move_time = 30.0  # Initial positioning
        
        total_time += move_time + point.count_time
        prev_angles = point.angles
    
    return DryRunResponse(
        valid=len(warnings) == 0,
        warnings=warnings,
        estimated_time=total_time,
        num_points=len(trajectory.points)
    )


@app.post("/config")
async def set_config(new_config: ProxyConfig):
    """Update proxy configuration."""
    global config, simulator, nice_connection, replay_data, replay_index
    
    config = new_config
    
    if config.mode == ProxyMode.SIMULATION:
        initialize_simulator()
    
    elif config.mode == ProxyMode.LIVE:
        if config.nice_host and config.nice_port:
            nice_connection = await connect_to_nice(
                config.nice_host, config.nice_port
            )
            if not nice_connection:
                return {"status": "warning", "message": "Failed to connect to NICE"}
    
    elif config.mode == ProxyMode.REPLAY:
        if config.replay_file:
            replay_data = load_replay_data(config.replay_file)
            replay_index = 0
    
    return {"status": "configured", "mode": config.mode.value}


@app.get("/history")
async def get_history():
    """Get measurement history (simulation mode only)."""
    if config.mode != ProxyMode.SIMULATION:
        raise HTTPException(
            status_code=400, 
            detail="History only available in simulation mode"
        )
    
    history = []
    for result in simulator.measurement_history:
        history.append({
            'h': result.point.h,
            'k': result.point.k,
            'l': result.point.l,
            'E': result.point.E,
            'intensity': result.intensity,
            'uncertainty': result.uncertainty,
            'elapsed_time': result.elapsed_time,
            'timestamp': result.point.timestamp
        })
    
    return {
        'total_measurements': len(history),
        'total_time': simulator.total_simulated_time,
        'measurements': history
    }


@app.post("/reset")
async def reset():
    """Reset simulator state (simulation mode only)."""
    if config.mode == ProxyMode.SIMULATION:
        simulator.reset()
        return {"status": "reset"}
    else:
        raise HTTPException(
            status_code=400,
            detail="Reset only available in simulation mode"
        )


# =============================================================================
# Helper functions
# =============================================================================

def estimate_move_time(from_angles: Dict[str, float], 
                       to_angles: Dict[str, float]) -> float:
    """Estimate time to move between angle configurations."""
    max_delta = 0
    for key in from_angles:
        if key in to_angles and key.startswith('A'):
            delta = abs(to_angles[key] - from_angles[key])
            max_delta = max(max_delta, delta)
    
    # Assume 2 deg/sec average speed + 5s overhead
    return max_delta / 2.0 + 5.0


def load_replay_data(filepath: str) -> List[dict]:
    """Load recorded experiment data for replay."""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)


def get_next_replay_result() -> MeasureResponse:
    """Get next result from replay data."""
    global replay_index
    
    if replay_data is None or replay_index >= len(replay_data):
        raise HTTPException(status_code=404, detail="No more replay data")
    
    data = replay_data[replay_index]
    replay_index += 1
    
    return MeasureResponse(
        counts=data['counts'],
        monitor_counts=data.get('monitor_counts'),
        intensity=data['intensity'],
        uncertainty=data['uncertainty'],
        timestamp=time.time(),
        metadata={'replayed': True, 'original_timestamp': data.get('timestamp')}
    )


async def connect_to_nice(host: str, port: int):
    """
    Establish connection to NICE server.
    
    This would implement the actual NICE protocol.
    For now, returns None (not implemented).
    """
    logger.warning("NICE connection not implemented - use simulation mode")
    return None


async def query_nice_position():
    """Query current position from NICE."""
    # Would implement NICE protocol
    raise NotImplementedError("NICE integration not yet implemented")


async def send_nice_measurement(request: MeasureRequest):
    """Send measurement command to NICE."""
    # Would implement NICE protocol
    raise NotImplementedError("NICE integration not yet implemented")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Run the proxy server."""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="TAS-AI NICE Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--mode", default="simulation", 
                        choices=["live", "simulation", "replay", "passthrough"],
                        help="Operating mode")
    parser.add_argument("--nice-host", help="NICE server host (for live mode)")
    parser.add_argument("--nice-port", type=int, help="NICE server port")
    parser.add_argument("--replay-file", help="Replay data file (for replay mode)")
    
    args = parser.parse_args()
    
    # Update config from command line
    config.mode = ProxyMode(args.mode)
    config.nice_host = args.nice_host
    config.nice_port = args.nice_port
    config.replay_file = args.replay_file
    
    logger.info(f"Starting proxy server in {config.mode} mode")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
