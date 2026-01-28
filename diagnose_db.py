
import psycopg2
import json
import os
from psycopg2.extras import RealDictCursor

def load_config():
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def diagnose():
    config = load_config()
    print(f"Config Loaded: Host={config.get('db_host')}, DB={config.get('db_name')}, User={config.get('db_user')}")
    
    try:
        conn = psycopg2.connect(
            host=config.get("db_host", "localhost"),
            port=config.get("db_port", "5432"),
            user=config.get("db_user", "gsdh"),
            password=config.get("db_password", "123gsdh"),
            database=config.get("db_name", "gsdh"),
            connect_timeout=5
        )
        cur = conn.cursor()
        
        # 1. List Tables
        print("\n=== Tables in 'public' schema ===")
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()
        for t in tables:
            print(f"- {t[0]}")
            
        # 2. Detail Columns for our tables
        target_tables = ['gsdh_data', 'checkin_info']
        for table in target_tables:
            print(f"\n=== Columns for table '{table}' ===")
            cur.execute("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = %s
            """, (table,))
            columns = cur.fetchall()
            if not columns:
                print("(Table not found)")
            else:
                for col in columns:
                    print(f"  - {col[0]}: {col[1]}")

        conn.close()
        
    except Exception as e:
        print(f"\nConnection Failed: {e}")

if __name__ == "__main__":
    diagnose()
