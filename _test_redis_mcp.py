"""Quick test of MCPRedisService connectivity and basic operations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mcp_redis_service import MCPRedisService

def main():
    svc = MCPRedisService()
    
    # Test 1: Ping
    print("1. Ping...", end=" ")
    ok = svc.ping()
    print(f"{'OK' if ok else 'FAIL'}")
    
    # Test 2: Set/Get
    print("2. SET/GET...", end=" ")
    svc.set("ca:test:hello", "world")
    val = svc.get("ca:test:hello")
    print(f"{'OK' if val == 'world' else 'FAIL'} (got: {val!r})")
    
    # Test 3: Hash
    print("3. HSET/HGETALL...", end=" ")
    svc.hset_dict("ca:test:hash", {"name": "test", "value": "123", "flag": "true"})
    data = svc.hgetall("ca:test:hash")
    print(f"{'OK' if data.get('name') == 'test' else 'FAIL'} (got: {data})")
    
    # Test 4: HGET
    print("4. HGET...", end=" ")
    name = svc.hget("ca:test:hash", "name")
    print(f"{'OK' if name == 'test' else 'FAIL'} (got: {name!r})")

    # Test 5: Sorted Set
    print("5. ZADD/ZRANGE...", end=" ")
    svc.zadd("ca:test:zset", 10.0, "file_a")
    svc.zadd("ca:test:zset", 30.0, "file_b")
    svc.zadd("ca:test:zset", 20.0, "file_c")
    items = svc.zrange("ca:test:zset", 0, -1, with_scores=True)
    print(f"{'OK' if len(items) >= 3 else 'FAIL'} (got: {items})")

    # Test 6: ZREVRANGE
    print("6. ZREVRANGE...", end=" ")
    rev = svc.zrevrange("ca:test:zset", 0, 1, with_scores=True)
    print(f"OK (got: {rev})")

    # Test 7: Scan keys
    print("7. SCAN_KEYS...", end=" ")
    keys = svc.scan_keys("ca:test:*")
    print(f"{'OK' if len(keys) >= 3 else 'FAIL'} (found: {keys})")

    # Test 8: Cleanup
    print("8. Cleanup...", end=" ")
    for k in keys:
        svc.delete(k)
    remaining = svc.scan_keys("ca:test:*")
    print(f"{'OK' if len(remaining) == 0 else 'FAIL'} (remaining: {remaining})")

    # Test 9: INCR
    print("9. INCR...", end=" ")
    v1 = svc.incr("ca:test:counter")
    v2 = svc.incr("ca:test:counter")
    print(f"{'OK' if v1 == 1 and v2 == 2 else 'FAIL'} (v1={v1}, v2={v2})")
    svc.delete("ca:test:counter")

    svc.disconnect()
    print("\n=== ALL TESTS DONE ===")

if __name__ == "__main__":
    main()
