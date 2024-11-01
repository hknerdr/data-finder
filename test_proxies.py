# File: test_proxies.py

import json
import requests
from requests.exceptions import ProxyError, ConnectTimeout, SSLError

def load_proxies(config_path="config/proxy_config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('proxies', [])

def test_proxy(proxy):
    proxy_url = f"{proxy['protocol']}://{proxy['username']}:{proxy['password']}@{proxy['host']}:{proxy['port']}"
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    try:
        response = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=10)
        if response.status_code == 200:
            print(f"Proxy {proxy_url} is working. Response IP: {response.json()['origin']}")
        else:
            print(f"Proxy {proxy_url} returned status code {response.status_code}")
    except (ProxyError, ConnectTimeout, SSLError) as e:
        print(f"Proxy {proxy_url} failed with error: {e}")

def main():
    proxies = load_proxies()
    if not proxies:
        print("No proxies found in proxy_config.json.")
        return
    
    for proxy in proxies:
        test_proxy(proxy)

if __name__ == "__main__":
    main()
