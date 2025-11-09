from demo import DatabaseWithCache
import time

db = DatabaseWithCache()

post_id = 1

def reset():
    db.redis_client.delete(f"views:post:{post_id}")
    db.cursor.execute("UPDATE users SET views = 0 WHERE id = ?", (post_id,))
    db.conn.commit()

# Correctness demo (prints 1..25 and syncs at 10/20)
reset()
print("=== Testing View Counter ===")
for i in range(25):
    count = db.increment_post_views(post_id)
    print(f"View #{i+1}: Total views = {count}")

# Performance demo: Naive DB-only vs Redis-backed
N = 3000

# Naive DB-only (commit every view)
reset()
t0 = time.time()
for _ in range(N):
    db.cursor.execute("UPDATE users SET views = views + 1 WHERE id = ?", (post_id,))
    db.conn.commit()
db_only = time.time() - t0

# Redis-backed (DB sync every 10)
reset()
t0 = time.time()
for _ in range(N):
    db.increment_post_views(post_id)
redis_time = time.time() - t0

print(f"\nDB-only time: {db_only:.3f}s")
print(f"Redis time:   {redis_time:.3f}s")
print(f"Speedup:      {db_only/redis_time:.1f}x")

db.close()