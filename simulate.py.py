from demo import DatabaseWithCache
db = DatabaseWithCache()

post_id = 1
# reset Redis + DB
db.redis_client.delete(f"views:post:{post_id}")
db.cursor.execute("UPDATE users SET views = 0 WHERE id = ?", (post_id,))
db.conn.commit()

print("=== Testing View Counter ===")
for i in range(25):
    count = db.increment_post_views(post_id)
    print(f"View #{i+1}: Total views = {count}")
db.close()