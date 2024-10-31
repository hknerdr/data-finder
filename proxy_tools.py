from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import logging
import statistics
from typing import Dict, List, Optional
import threading
import time
import json
from pathlib import Path

@dataclass
class ProxyMetrics:
    success_count: int = 0
    failure_count: int = 0
    total_response_time: float = 0
    response_times: List[float] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
    
    @property
    def average_response_time(self) -> float:
        if not self.response_times:
            return 0
        return statistics.mean(self.response_times)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return (self.success_count / total * 100) if total > 0 else 0

class ProxyTester:
    def __init__(self, config_path: str = "config/proxy_config.json"):
        self.config_path = Path(config_path)
        self.metrics: Dict[str, ProxyMetrics] = {}
        self.lock = threading.Lock()
        self.test_urls = [
            "https://www.google.com",
            "https://www.microsoft.com",
            "https://www.amazon.com"
        ]
    
    def test_proxy(self, proxy_config: ProxyConfig) -> bool:
        proxy_url = proxy_config.get_url()
        if proxy_url not in self.metrics:
            self.metrics[proxy_url] = ProxyMetrics()
        
        success = False
        start_time = time.time()
        
        try:
            for test_url in self.test_urls:
                response = requests.get(
                    test_url,
                    proxies=proxy_config.to_dict(),
                    timeout=10
                )
                response.raise_for_status()
            
            response_time = time.time() - start_time
            with self.lock:
                self.metrics[proxy_url].success_count += 1
                self.metrics[proxy_url].response_times.append(response_time)
                self.metrics[proxy_url].total_response_time += response_time
                self.metrics[proxy_url].last_success = datetime.now()
            success = True
            
        except Exception as e:
            with self.lock:
                self.metrics[proxy_url].failure_count += 1
                self.metrics[proxy_url].last_failure = datetime.now()
            logging.error(f"Proxy test failed for {proxy_url}: {str(e)}")
        
        return success
    
    def test_all_proxies(self) -> Dict[str, Dict]:
        results = {}
        config_manager = ProxyConfigManager()
        
        for proxy in config_manager.proxies:
            success = self.test_proxy(proxy)
            metrics = self.metrics[proxy.get_url()]
            
            results[proxy.get_url()] = {
                'success': success,
                'success_rate': metrics.success_rate,
                'avg_response_time': metrics.average_response_time,
                'last_success': metrics.last_success,
                'last_failure': metrics.last_failure
            }
        
        return results

class ProxyMonitor:
    def __init__(self):
        self.metrics: Dict[str, ProxyMetrics] = {}
        self.lock = threading.Lock()
    
    def record_success(self, proxy_url: str, response_time: float):
        with self.lock:
            if proxy_url not in self.metrics:
                self.metrics[proxy_url] = ProxyMetrics()
            
            metrics = self.metrics[proxy_url]
            metrics.success_count += 1
            metrics.response_times.append(response_time)
            metrics.last_success = datetime.now()
    
    def record_failure(self, proxy_url: str):
        with self.lock:
            if proxy_url not in self.metrics:
                self.metrics[proxy_url] = ProxyMetrics()
            
            metrics = self.metrics[proxy_url]
            metrics.failure_count += 1
            metrics.last_failure = datetime.now()
    
    def get_metrics(self) -> Dict[str, Dict]:
        with self.lock:
            return {
                proxy_url: {
                    'success_rate': metrics.success_rate,
                    'avg_response_time': metrics.average_response_time,
                    'success_count': metrics.success_count,
                    'failure_count': metrics.failure_count,
                    'last_success': metrics.last_success,
                    'last_failure': metrics.last_failure
                }
                for proxy_url, metrics in self.metrics.items()
            }