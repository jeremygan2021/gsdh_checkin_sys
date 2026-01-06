from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict
import random
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Database connection parameters
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "121.43.104.161"),
    "port": os.getenv("DB_PORT", "6432"),
    "user": os.getenv("DB_USER", "gsdh"),
    "password": os.getenv("DB_PASSWORD", "123gsdh"),
    "database": os.getenv("DB_NAME", "gsdh")
}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Models
class CheckinRequest(BaseModel):
    gsdh_id: str
    name: str
    phone: str
    company_name: Optional[str] = None
    position: Optional[str] = None
    business_scope: Optional[str] = None
    vision_2026: Optional[str] = None
    location: Optional[str] = None

class AddUserRequest(BaseModel):
    name: str
    phone: str
    industry_company: Optional[str] = None
    fee: Optional[str] = None
    payment_channel: Optional[str] = None

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

def assign_seat(cur, user_industry: str) -> str:
    """
    Allocate a seat based on:
    1. Even distribution (13 tables, max 12 per table)
    2. Mix industries (try to put user in a table where their industry is least represented)
    """
    TOTAL_TABLES = 13
    MAX_PER_TABLE = 12
    
    # Initialize table stats
    # tables = { 1: {'count': 0, 'industries': []}, ... }
    tables = {i: {'count': 0, 'industries': []} for i in range(1, TOTAL_TABLES + 1)}
    
    # Fetch current seating status
    # We join checkin_info with gsdh_data to get industries of people ALREADY SEATED
    query = """
    SELECT ci.location, gd.industry_company 
    FROM checkin_info ci 
    JOIN gsdh_data gd ON ci.gsdh_id = gd.new_id 
    WHERE ci.location IS NOT NULL AND ci.location LIKE '第%桌'
    """
    cur.execute(query)
    rows = cur.fetchall()
    
    for row in rows:
        loc = row[0] # e.g. "第1桌"
        ind = row[1]
        try:
            # Extract table number
            table_num = int(loc.replace("第", "").replace("桌", ""))
            if 1 <= table_num <= TOTAL_TABLES:
                tables[table_num]['count'] += 1
                if ind:
                    tables[table_num]['industries'].append(ind)
        except ValueError:
            continue
            
    # Filter tables that are not full
    available_tables = [t for t in tables.items() if t[1]['count'] < MAX_PER_TABLE]
    
    if not available_tables:
        return "自由席" # Fallback if all full
        
    # Strategy 1: Find tables with Minimum Count (Even Distribution)
    min_count = min(t[1]['count'] for t in available_tables)
    candidates_step1 = [t for t in available_tables if t[1]['count'] == min_count]
    
    # Strategy 2: Among candidates, find best for diversity
    # We want a table where user_industry is NOT present, or present least often
    best_table = None
    
    # If user has no industry info, just pick random from candidates
    if not user_industry:
        best_table = random.choice(candidates_step1)[0]
    else:
        # Score candidates: lower score is better (score = count of this industry in that table)
        scored_candidates = []
        for table_id, stats in candidates_step1:
            # Simple fuzzy check: count how many times user_industry appears in stats['industries']
            # We use simple string containment
            industry_count = sum(1 for existing_ind in stats['industries'] if existing_ind and user_industry in existing_ind)
            scored_candidates.append((table_id, industry_count))
        
        # Sort by industry count (asc)
        scored_candidates.sort(key=lambda x: x[1])
        
        # Pick the one with least collision
        min_collision = scored_candidates[0][1]
        final_candidates = [x[0] for x in scored_candidates if x[1] == min_collision]
        best_table = random.choice(final_candidates)
        
    return f"第{best_table}桌"

def get_tablemates(cur, location: str, exclude_id: str) -> List[Dict]:
    """
    Get up to 3 random tablemates from the same table location.
    """
    if not location or location == "自由席":
        return []
    
    # Debug: Check who is at this location
    print(f"DEBUG: Fetching tablemates for location: '{location}', excluding: '{exclude_id}'")
    
    # Important: Ensure the location string format matches database exactly
    # Database seems to store "第X桌", ensuring consistent querying
    
    # Updated query to fetch more details from checkin_info
    query = """
    SELECT ci.name, gd.industry_company, ci.company_name, ci.position, ci.business_scope, ci.vision_2026
    FROM checkin_info ci
    LEFT JOIN gsdh_data gd ON ci.gsdh_id = gd.new_id
    WHERE ci.location = %s AND ci.gsdh_id != %s
    ORDER BY RANDOM()
    LIMIT 3
    """
    cur.execute(query, (location, exclude_id))
    rows = cur.fetchall()
    
    print(f"DEBUG: Found {len(rows)} tablemates")
    
    tablemates = []
    for row in rows:
        tablemates.append({
            "name": row[0],
            "industry": row[1] or "暂无行业信息",
            "company_name": row[2] or "暂无单位信息",
            "position": row[3] or "暂无职务信息",
            "business_scope": row[4] or "暂无业务信息",
            "vision_2026": row[5] or "暂无愿景信息"
        })
    return tablemates

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/search")
async def search_user(query: str):
    """
    Search user by phone (exact match) or name (fuzzy match).
    """
    print(f"DEBUG: Searching for query: {query}")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Priority 1: Exact Phone Match
        cur.execute("SELECT * FROM gsdh_data WHERE phone = %s", (query,))
        user = cur.fetchone()
        
        # Priority 2: Fuzzy Name Match if not found by phone
        if not user:
            # Using ILIKE for case-insensitive fuzzy search
            cur.execute("SELECT * FROM gsdh_data WHERE name ILIKE %s", (f"%{query}%",))
            users = cur.fetchall()
            
            if len(users) == 0:
                conn.close()
                return JSONResponse(content={"found": False, "message": "未查询到相关信息，请检查输入是否正确"}, status_code=404)
            elif len(users) > 1:
                # If multiple users found by name, return list for user to select (simplified here to return first or error)
                # For this MVP, let's return all matching users so frontend can handle selection
                conn.close()
                return JSONResponse(content={"found": True, "multiple": True, "users": users})
            else:
                user = users[0]
        
        # Check if already signed
        if user.get('is_signed') == 'TRUE':
             # If already signed, fetch their assigned seat and tablemates
             cur.execute("SELECT location FROM checkin_info WHERE gsdh_id = %s", (user['new_id'],))
             checkin_info = cur.fetchone()
             assigned_seat = checkin_info['location'] if checkin_info else "自由席"
             
             # Fetch tablemates
             # Use a new cursor for the helper function to avoid cursor factory conflict or state issues
             # Note: get_tablemates expects a standard cursor for tuple results, but here we have DictCursor
             # We can adapt get_tablemates or just use key access if we pass the DictCursor
             # Let's create a fresh standard cursor to be safe and consistent with get_tablemates implementation
             cur_plain = conn.cursor() 
             tablemates = get_tablemates(cur_plain, assigned_seat, user['new_id'])
             cur_plain.close()
             
             conn.close()
             return JSONResponse(content={
                 "found": True, 
                 "user": user, 
                 "already_signed": True, 
                 "seat": assigned_seat,
                 "tablemates": tablemates
             })
        
        conn.close()
        return JSONResponse(content={"found": True, "user": user, "already_signed": False})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/checkin")
async def checkin_user(checkin_data: CheckinRequest):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 0. Get user's industry from gsdh_data to help with seat allocation
        cur.execute("SELECT industry_company FROM gsdh_data WHERE new_id = %s", (checkin_data.gsdh_id,))
        res = cur.fetchone()
        user_industry = res[0] if res else ""
        
        # 1. Allocate Seat
        assigned_seat = assign_seat(cur, user_industry)
        
        # 2. Insert into checkin_info with assigned seat
        insert_sql = """
        INSERT INTO checkin_info 
        (name, phone, company_name, position, business_scope, vision_2026, location, gsdh_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_sql, (
            checkin_data.name,
            checkin_data.phone,
            checkin_data.company_name,
            checkin_data.position,
            checkin_data.business_scope,
            checkin_data.vision_2026,
            assigned_seat,  # Use the generated seat
            checkin_data.gsdh_id
        ))
        
        # 3. Update gsdh_data is_signed to TRUE
        update_sql = "UPDATE gsdh_data SET is_signed = 'TRUE' WHERE new_id = %s"
        cur.execute(update_sql, (checkin_data.gsdh_id,))
        
        conn.commit()
        
        # 4. Fetch tablemates for the newly assigned seat
        tablemates = get_tablemates(cur, assigned_seat, checkin_data.gsdh_id)
        
        cur.close()
        conn.close()
        
        return {"success": True, "message": "签到成功！", "seat": assigned_seat, "tablemates": tablemates}
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        return JSONResponse(content={"success": False, "message": f"签到失败: {str(e)}"}, status_code=500)

@app.get("/add-user", response_class=HTMLResponse)
async def add_user_page(request: Request):
    secret = os.getenv("ADD_USER_SECRET", "123quant-speed")
    print(f"DEBUG: Secret loaded: '{secret}'")
    return templates.TemplateResponse("add_user.html", {"request": request, "secret": secret})

@app.post("/api/add-user")
async def add_user_api(user_data: AddUserRequest):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if phone already exists
        cur.execute("SELECT * FROM gsdh_data WHERE phone = %s", (user_data.phone,))
        if cur.fetchone():
            conn.close()
            return JSONResponse(content={"success": False, "message": "该手机号已存在"}, status_code=400)
            
        # Calculate next new_id
        cur.execute("SELECT MAX(CAST(new_id AS INTEGER)) FROM gsdh_data WHERE new_id ~ '^[0-9]+$'")
        row = cur.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        new_id = str(max_id + 1)
        
        insert_sql = """
        INSERT INTO gsdh_data (new_id, name, phone, industry_company, fee, payment_channel, is_signed)
        VALUES (%s, %s, %s, %s, %s, %s, 'FALSE')
        """
        cur.execute(insert_sql, (
            new_id, 
            user_data.name, 
            user_data.phone, 
            user_data.industry_company,
            user_data.fee,
            user_data.payment_channel
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {"success": True, "message": "添加成功", "new_id": new_id}
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        return JSONResponse(content={"success": False, "message": f"添加失败: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
