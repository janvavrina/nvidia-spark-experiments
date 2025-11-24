"""
Performance profiling utilities for DGX Spark
Monitors NVLink-C2C bandwidth, unified memory usage, and system performance.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time"""
    timestamp: float
    cpu_memory_used: float  # GB
    cpu_memory_percent: float
    gpu_memory_used: Optional[float] = None  # GB
    gpu_memory_total: Optional[float] = None  # GB
    unified_memory_used: Optional[float] = None  # GB
    unified_memory_total: Optional[float] = None  # GB


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    duration: float
    memory_snapshots: List[MemorySnapshot]
    peak_cpu_memory: float
    peak_gpu_memory: Optional[float]
    avg_cpu_memory: float
    avg_gpu_memory: Optional[float]


class MemoryProfiler:
    """Profiles memory usage over time"""
    
    def __init__(self, interval: float = 0.1):
        """
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def _get_gpu_memory(self):
        """Get GPU memory usage in GB"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None, None
        
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
        except:
            return None, None
    
    def _get_unified_memory(self):
        """Get unified memory usage in GB"""
        # For Grace Blackwell, unified memory is accessible via CUDA
        # This is a placeholder - actual implementation depends on CUDA APIs
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None, None
        
        try:
            # Unified memory should be accessible via CUDA unified memory APIs
            # This is a simplified version
            total = 128.0  # 128GB unified memory on DGX Spark
            # Try to get actual usage - this may require specific CUDA calls
            return None, total
        except:
            return None, None
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            # CPU memory
            cpu_mem = psutil.virtual_memory()
            cpu_used_gb = cpu_mem.used / 1024**3
            cpu_percent = cpu_mem.percent
            
            # GPU memory
            gpu_allocated, gpu_reserved = self._get_gpu_memory()
            
            # Unified memory
            unified_used, unified_total = self._get_unified_memory()
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                cpu_memory_used=cpu_used_gb,
                cpu_memory_percent=cpu_percent,
                gpu_memory_used=gpu_allocated,
                gpu_memory_total=gpu_reserved,
                unified_memory_used=unified_used,
                unified_memory_total=unified_total,
            )
            
            self.snapshots.append(snapshot)
            time.sleep(self.interval)
    
    def start(self):
        """Start memory profiling"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self) -> PerformanceMetrics:
        """Stop memory profiling and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.snapshots:
            return PerformanceMetrics(
                duration=0.0,
                memory_snapshots=[],
                peak_cpu_memory=0.0,
                peak_gpu_memory=None,
                avg_cpu_memory=0.0,
                avg_gpu_memory=None,
            )
        
        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        
        cpu_memories = [s.cpu_memory_used for s in self.snapshots]
        gpu_memories = [s.gpu_memory_used for s in self.snapshots if s.gpu_memory_used is not None]
        
        return PerformanceMetrics(
            duration=duration,
            memory_snapshots=self.snapshots,
            peak_cpu_memory=max(cpu_memories),
            peak_gpu_memory=max(gpu_memories) if gpu_memories else None,
            avg_cpu_memory=sum(cpu_memories) / len(cpu_memories),
            avg_gpu_memory=sum(gpu_memories) / len(gpu_memories) if gpu_memories else None,
        )
    
    def get_current_memory(self) -> Dict:
        """Get current memory usage"""
        cpu_mem = psutil.virtual_memory()
        gpu_allocated, gpu_reserved = self._get_gpu_memory()
        unified_used, unified_total = self._get_unified_memory()
        
        return {
            "cpu_memory_used_gb": cpu_mem.used / 1024**3,
            "cpu_memory_percent": cpu_mem.percent,
            "gpu_memory_allocated_gb": gpu_allocated,
            "gpu_memory_reserved_gb": gpu_reserved,
            "unified_memory_used_gb": unified_used,
            "unified_memory_total_gb": unified_total,
        }


class NVLinkProfiler:
    """Profiles NVLink-C2C bandwidth and connectivity"""
    
    def __init__(self):
        self.initialized = False
        if NVML_AVAILABLE:
            try:
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.initialized = True
            except:
                pass
    
    def get_nvlink_topology(self) -> Dict:
        """Get NVLink topology information"""
        if not self.initialized:
            return {"error": "NVML not available"}
        
        topology = {}
        try:
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # Get NVLink information
                # This requires specific NVML APIs for NVLink
                topology[f"device_{i}"] = {
                    "name": pynvml.nvmlDeviceGetName(handle).decode(),
                }
        except Exception as e:
            topology["error"] = str(e)
        
        return topology
    
    def get_bandwidth_stats(self) -> Dict:
        """Get NVLink bandwidth statistics"""
        # This would require specific CUDA/NVML APIs
        # Placeholder for actual implementation
        return {
            "note": "NVLink-C2C bandwidth monitoring requires specific CUDA APIs",
            "unified_memory_bandwidth": "N/A",
        }


def profile_function(func, *args, **kwargs) -> tuple[any, PerformanceMetrics]:
    """
    Profile a function's execution and memory usage
    
    Returns:
        (result, metrics): Function result and performance metrics
    """
    profiler = MemoryProfiler()
    profiler.start()
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
    finally:
        metrics = profiler.stop()
        end_time = time.time()
        metrics.duration = end_time - start_time
    
    return result, metrics


def save_metrics(metrics: PerformanceMetrics, filepath: str):
    """Save performance metrics to JSON file"""
    data = {
        "duration": metrics.duration,
        "peak_cpu_memory_gb": metrics.peak_cpu_memory,
        "peak_gpu_memory_gb": metrics.peak_gpu_memory,
        "avg_cpu_memory_gb": metrics.avg_cpu_memory,
        "avg_gpu_memory_gb": metrics.avg_gpu_memory,
        "snapshots": [
            {
                "timestamp": s.timestamp,
                "cpu_memory_used_gb": s.cpu_memory_used,
                "cpu_memory_percent": s.cpu_memory_percent,
                "gpu_memory_used_gb": s.gpu_memory_used,
                "gpu_memory_total_gb": s.gpu_memory_total,
                "unified_memory_used_gb": s.unified_memory_used,
                "unified_memory_total_gb": s.unified_memory_total,
            }
            for s in metrics.memory_snapshots
        ],
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

