import os
import psycopg2
from dotenv import load_dotenv

# 模拟 main.py 的行为：加载环境变量
load_dotenv()

# 使用代码中的默认配置（模拟服务器环境），但会被 .env 覆盖（模拟本地环境）
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "gsdh"),
    "password": os.getenv("DB_PASSWORD", "123gsdh"),
    "database": os.getenv("DB_NAME", "gsdh")
}

print(f"Connecting to: {DB_CONFIG['host']}:{DB_CONFIG['port']}")

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("Successfully connected to the database!")
    conn.close()
except Exception as e:
    print(f"Failed to connect: {e}")
