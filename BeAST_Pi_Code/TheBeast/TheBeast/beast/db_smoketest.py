import os
from dotenv import load_dotenv
import psycopg2

# Load .env file
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "beast")
DB_USER = os.getenv("DB_USER", "beast")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

print(f"Connecting to {DB_NAME} at {DB_HOST}:{DB_PORT} as {DB_USER}...")

conn = psycopg2.connect(
    host=DB_HOST,
    port=int(DB_PORT),
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)

cur = conn.cursor()

# Show a few raw tables
cur.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema IN ('raw', 'agg')
    ORDER BY table_schema, table_name
    LIMIT 10;
""")
rows = cur.fetchall()
print("✅ Tables found:")
for schema, name in rows:
    print(f"  {schema}.{name}")

cur.close()
conn.close()
print("✅ DB smoke test OK")
