"""
NICE Proxy Client

Connects to a proxy server that mediates communication with NICE.
The proxy server can operate in different modes:
- live: Forward commands to real NICE
- simulation: Use internal simulator
- replay: Replay recorded experiments
"""

import requests
import logging
import time
from typing import Optional, Callable, Dict
from dataclasses import asdict

from .base import (
    InstrumentInterface, MeasurementPoint, MeasurementResult, TASGeometry
)

logger = logging.getLogger(__name__)


class NICEProxyClient(InstrumentInterface):
    """
    Client for NICE proxy server.
    
    This connects to a proxy server (see proxy_server/) that handles
    actual NICE communication. This separation allows:
    
    1. Running TAS-AI on a separate machine from instrument control
    2. Injecting simulated data for testing
    3. Recording and replaying experiments
    4. Adding safety checks before commands reach NICE
    5. Human-in-the-loop approval workflows
    """
    
    def __init__(self,
                 proxy_url: str = "http://localhost:8080",
                 geometry: Optional[TASGeometry] = None,
                 timeout: float = 30.0,
                 dry_run: bool = False):
        """
        Initialize NICE proxy client.
        
        Parameters
        ----------
        proxy_url : str
            URL of the proxy server
        geometry : TASGeometry, optional
            Local geometry calculator (if None, proxy calculates angles)
        timeout : float
            Request timeout in seconds
        dry_run : bool
            If True, don't execute measurements (for testing)
        """
        self.proxy_url = proxy_url.rstrip('/')
        self.geometry = geometry
        self.timeout = timeout
        self.dry_run = dry_run
        self._connected = False
        
        # Callbacks for human-in-the-loop
        self._approval_callback: Optional[Callable[[MeasurementPoint], bool]] = None
        self._status_callback: Optional[Callable[[str], None]] = None
        self._error_callback: Optional[Callable[[str, Exception], None]] = None
    
    # =========================================================================
    # Callback registration
    # =========================================================================
    
    def set_approval_callback(self, callback: Callable[[MeasurementPoint], bool]):
        """
        Set callback for human approval before measurements.
        
        The callback receives the MeasurementPoint and should return
        True to approve or False to reject.
        """
        self._approval_callback = callback
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates."""
        self._status_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, Exception], None]):
        """Set callback for error handling."""
        self._error_callback = callback
    
    def _notify_status(self, message: str):
        """Send status update to callback."""
        logger.info(message)
        if self._status_callback:
            self._status_callback(message)
    
    def _notify_error(self, message: str, exception: Exception):
        """Send error to callback."""
        logger.error(f"{message}: {exception}")
        if self._error_callback:
            self._error_callback(message, exception)
    
    # =========================================================================
    # Connection management
    # =========================================================================
    
    def connect(self) -> bool:
        """Establish connection to proxy server."""
        try:
            response = requests.get(
                f"{self.proxy_url}/status",
                timeout=5
            )
            if response.status_code == 200:
                self._connected = True
                status = response.json()
                self._notify_status(
                    f"Connected to NICE proxy at {self.proxy_url} "
                    f"(mode: {status.get('mode', 'unknown')})"
                )
                return True
            else:
                self._notify_error(
                    "Proxy returned error",
                    Exception(f"Status code: {response.status_code}")
                )
                return False
        except requests.RequestException as e:
            self._notify_error("Failed to connect to proxy", e)
            return False
    
    def disconnect(self):
        """Disconnect from proxy server."""
        self._connected = False
        self._notify_status("Disconnected from proxy")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    # =========================================================================
    # InstrumentInterface implementation
    # =========================================================================
    
    def get_current_position(self) -> MeasurementPoint:
        """Query current instrument position from proxy."""
        response = requests.get(
            f"{self.proxy_url}/position",
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return MeasurementPoint(
            h=data['h'],
            k=data['k'],
            l=data['l'],
            E=data['E'],
            angles=data.get('angles')
        )
    
    def calculate_angles(self, point: MeasurementPoint) -> Dict[str, float]:
        """
        Calculate instrument angles for (h,k,l,E).
        
        Uses local geometry if available, otherwise asks proxy.
        """
        if self.geometry:
            return self.geometry.hkl_to_angles(point.h, point.k, point.l, point.E)
        
        # Ask proxy to calculate
        response = requests.post(
            f"{self.proxy_url}/calculate_angles",
            json={'h': point.h, 'k': point.k, 'l': point.l, 'E': point.E},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def estimate_move_time(self, from_point: MeasurementPoint,
                           to_point: MeasurementPoint) -> float:
        """Estimate movement time between points."""
        # Calculate angles if needed
        if from_point.angles is None:
            try:
                from_point.angles = self.calculate_angles(from_point)
            except Exception:
                pass
        
        if to_point.angles is None:
            try:
                to_point.angles = self.calculate_angles(to_point)
            except Exception:
                pass
        
        if from_point.angles and to_point.angles:
            # Simple model: assume ~2 deg/sec average, 5s overhead
            max_delta = 0
            for key in from_point.angles:
                if key in to_point.angles and key.startswith('A'):
                    delta = abs(to_point.angles[key] - from_point.angles[key])
                    max_delta = max(max_delta, delta)
            return max_delta / 2.0 + 5.0
        
        return 30.0  # Default estimate
    
    def validate_point(self, point: MeasurementPoint) -> tuple:
        """Validate if measurement point is achievable."""
        # Check with local geometry first if available
        if self.geometry:
            if not self.geometry.is_accessible(point.h, point.k, point.l, point.E):
                return False, "Point not accessible (local geometry check)"
        
        # Also validate with proxy (may have more constraints)
        try:
            response = requests.post(
                f"{self.proxy_url}/validate",
                json={'h': point.h, 'k': point.k, 'l': point.l, 'E': point.E},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data['valid'], data.get('reason', '')
        except requests.RequestException as e:
            logger.warning(f"Proxy validation failed, using local only: {e}")
            return True, ""
    
    def measure(self, point: MeasurementPoint) -> MeasurementResult:
        """
        Execute measurement via proxy.
        
        Implements full human-in-the-loop workflow:
        1. Validate point
        2. Calculate angles
        3. Request human approval (if callback set)
        4. Execute measurement (or dry run)
        5. Return result
        """
        start_time = time.time()
        
        # Step 1: Validate
        valid, reason = self.validate_point(point)
        if not valid:
            raise ValueError(f"Invalid measurement point: {reason}")
        
        # Step 2: Calculate angles
        point.angles = self.calculate_angles(point)
        
        # Step 3: Human approval
        if self._approval_callback:
            self._notify_status(
                f"Requesting approval for ({point.h:.3f}, {point.k:.3f}, "
                f"{point.l:.3f}, {point.E:.2f} meV), {point.count_time}s"
            )
            approved = self._approval_callback(point)
            if not approved:
                raise RuntimeError("Measurement rejected by operator")
        
        # Step 4: Status update
        self._notify_status(
            f"Measuring at ({point.h:.3f}, {point.k:.3f}, {point.l:.3f}, "
            f"{point.E:.2f} meV)"
        )
        
        # Step 5: Execute or dry run
        if self.dry_run:
            logger.info(f"DRY RUN: Would measure at {point}")
            return MeasurementResult(
                point=point,
                intensity=0.0,
                uncertainty=0.0,
                elapsed_time=point.count_time + 30.0,
                metadata={'dry_run': True}
            )
        
        # Send to proxy
        try:
            response = requests.post(
                f"{self.proxy_url}/measure",
                json={
                    'h': point.h,
                    'k': point.k,
                    'l': point.l,
                    'E': point.E,
                    'angles': point.angles,
                    'count_time': point.count_time,
                    'monitor': point.monitor
                },
                timeout=self.timeout + point.count_time + 120  # Extra time for movement
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self._notify_error("Measurement request failed", e)
            raise
        
        data = response.json()
        elapsed_time = time.time() - start_time
        
        # Update point with results
        point.counts = data['counts']
        point.monitor_counts = data.get('monitor_counts')
        point.timestamp = data.get('timestamp', time.time())
        
        result = MeasurementResult(
            point=point,
            intensity=data['intensity'],
            uncertainty=data['uncertainty'],
            elapsed_time=elapsed_time,
            metadata=data.get('metadata', {})
        )
        
        self._notify_status(
            f"Measurement complete: I={result.intensity:.2f}±{result.uncertainty:.2f}"
        )
        
        return result
    
    # =========================================================================
    # Trajectory operations
    # =========================================================================
    
    def generate_trajectory(self, points: list) -> dict:
        """Generate NICE trajectory for multiple points."""
        trajectory = {
            'type': 'TAS_SCAN',
            'points': []
        }
        
        for pt in points:
            if pt.angles is None:
                pt.angles = self.calculate_angles(pt)
            trajectory['points'].append({
                'h': pt.h, 'k': pt.k, 'l': pt.l, 'E': pt.E,
                'angles': pt.angles,
                'count_time': pt.count_time
            })
        
        return trajectory
    
    def dry_run_trajectory(self, trajectory: dict) -> dict:
        """
        Send trajectory to proxy for dry-run validation.
        
        Returns estimated times, warnings, etc.
        """
        response = requests.post(
            f"{self.proxy_url}/trajectory/dryrun",
            json=trajectory,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def execute_trajectory(self, trajectory: dict) -> list:
        """
        Execute a pre-built trajectory.
        
        Returns list of MeasurementResult.
        """
        # Validate first
        validation = self.dry_run_trajectory(trajectory)
        if not validation.get('valid', False):
            warnings = validation.get('warnings', [])
            raise ValueError(f"Trajectory validation failed: {warnings}")
        
        results = []
        for point_dict in trajectory['points']:
            point = MeasurementPoint(
                h=point_dict['h'],
                k=point_dict['k'],
                l=point_dict['l'],
                E=point_dict['E'],
                angles=point_dict.get('angles'),
                count_time=point_dict.get('count_time', 60.0)
            )
            result = self.measure(point)
            results.append(result)
        
        return results
    
    # =========================================================================
    # Proxy configuration
    # =========================================================================
    
    def set_proxy_mode(self, mode: str, **kwargs) -> dict:
        """
        Configure proxy server mode.
        
        Parameters
        ----------
        mode : str
            'live', 'simulation', or 'replay'
        **kwargs
            Additional mode-specific configuration
        """
        config = {'mode': mode, **kwargs}
        response = requests.post(
            f"{self.proxy_url}/config",
            json=config,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_proxy_status(self) -> dict:
        """Get current proxy server status."""
        response = requests.get(
            f"{self.proxy_url}/status",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
