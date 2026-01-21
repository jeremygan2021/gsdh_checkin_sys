from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict
import random
import uuid
import os
import json
import shutil
import difflib
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# Config Management
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "event_title": "云南AI共生大会",
    "event_sub_title": "2026 INTELLIGENT LEADERSHIP • AI SYMBIOSIS",
    "event_time": "1月10日 下午 2:00",
    "event_location": "金鼎科技园18号平台B座一楼报告厅",
    "event_content": "邀请重磅大咖分享AI在各行业的企业应用及案例，含深度交流环节。",
    "primary_color": "#00f2ff",
    "secondary_color": "#0066ff",
    "bg_color": "#050814",
    "header_image": "/static/image.jpg",
    "db_host": os.getenv("DB_HOST", "localhost"),
    "db_port": os.getenv("DB_PORT", "5432"),
    "db_user": os.getenv("DB_USER", "gsdh"),
    "db_password": os.getenv("DB_PASSWORD", "123gsdh"),
    "db_name": os.getenv("DB_NAME", "gsdh"),
    "enable_ticket_validation": True,
    "enable_seating": True,
    "total_tables": 14,
    "max_per_table": 10,
    "field_config": {
        "name": {"label": "姓名", "show": True, "required": True},
        "phone": {"label": "手机号码", "show": True, "required": True},
        "company_name": {"label": "单位名称", "show": True, "required": False},
        "position": {"label": "职务", "show": True, "required": False},
        "business_scope": {"label": "公司主要经营 / 业务", "show": True, "required": False},
        "vision_2026": {"label": "2026年业务愿景", "show": True, "required": False}
    },
    "wall_config": {
        "show_title": True,
        "learn_more_url": "https://www.example.com",
        "show_fields": {
             "name": True,
             "company_name": True,
             "position": True,
             "vision_2026": True,
             "business_scope": True
        },
        "bg_opacity": 0.3
    }
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            # Merge with default config to ensure all keys exist
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            
            # Deep merge for wall_config
            if "wall_config" in config and isinstance(config["wall_config"], dict):
                for k, v in DEFAULT_CONFIG["wall_config"].items():
                    if k not in config["wall_config"]:
                        config["wall_config"][k] = v
            elif "wall_config" not in config:
                 config["wall_config"] = DEFAULT_CONFIG["wall_config"]
                 
            return config
    return DEFAULT_CONFIG

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

CONFIG = load_config()

app = FastAPI()

# 进程池全局变量
process_pool = None

@app.on_event("startup")
async def startup_event():
    global process_pool
    # RK3588 有 8 个核心，预留一些给数据库和系统，使用 6 个核心进行计算
    process_pool = ProcessPoolExecutor(max_workers=6)
    print("ProcessPoolExecutor initialized with 6 workers")

@app.on_event("shutdown")
async def shutdown_event():
    if process_pool:
        process_pool.shutdown()
        print("ProcessPoolExecutor shutdown")

# Database connection parameters - Moved to CONFIG

# 商业领域同义词库 (Business Thesaurus) - 用于解决模糊语义匹配
BUSINESS_THESAURUS = {
    # 核心意图: [关联行业/关键词列表]
    "上市": ["IPO", "证券", "股票", "股份", "路演", "投行", "辅导", "财报", "合规", "董秘", "财务顾问", "审计", "律所", "金融", "机构"],
    "证券": ["上市", "交易", "股票", "投资", "金融", "资本", "券商", "投行"],
    "融资": ["找钱", "搞钱", "资金", "投资", "VC", "PE", "天使", "风投", "路演", "BP", "基金", "银行", "贷款"],
    "资金": ["融资", "投资", "银行", "贷款", "过桥", "保理", "供应链金融"],
    "获客": ["销售", "渠道", "推广", "流量", "代理", "分销", "增长", "营销", "广告", "传媒", "品牌", "私域"],
    "销售": ["获客", "渠道", "代理", "分销", "带货", "电商", "直播"],
    "技术": ["研发", "代码", "程序", "系统", "平台", "App", "小程序", "AI", "智能", "软件", "SaaS", "数字化", "算法", "架构"],
    "法律": ["合规", "律师", "法务", "合同", "知识产权", "维权", "纠纷", "仲裁", "数据合规"],
    "财税": ["会计", "审计", "报税", "记账", "财务", "税务", "节税"],
    "出海": ["跨境", "外贸", "物流", "海外", "国际", "通关", "Tiktok", "多语言", "本地化"],
    "供应链": ["物流", "仓储", "采购", "原材料", "制造", "工厂", "代工", "OEM"],
    "人力": ["招聘", "猎头", "培训", "HR", "劳务", "派遣", "灵活用工"],
    
    # 短视频与互联网专项扩展
    "自媒体": ["抖音", "快手", "视频号", "小红书", "B站", "直播", "带货", "种草", "网红", "KOL", "KOC", "MCN", "内容创作", "剪辑", "拍摄", "流量", "完播率", "点赞", "评论", "转发", "DOU+", "投流", "橱窗", "小黄车", "团购", "同城号", "剧情号", "知识号", "颜值号", "三农号"],
    "互联网": ["电商", "平台", "流量", "运营", "产品", "用户增长", "裂变", "留存", "转化", "GMV", "DAU", "MAU", "PV", "UV", "SEO", "SEM", "ASO", "投放", "拉新", "促活", "留存", "变现", "闭环", "私域", "公域", "矩阵", "账号", "内容", "社群", "小程序", "H5", "Web", "App", "iOS", "Android", "中台", "SaaS", "PaaS", "IaaS", "云原生", "微服务", "低代码", "零代码", "敏捷", "DevOps", "CI/CD"],
    "电商": ["购物", "订单", "支付", "物流", "仓储", "采购", "原材料", "制造", "工厂", "代工", "OEM"],
    # AI 行业专项扩展
    "AI": ["大模型", "算法", "算力", "芯片", "数据", "数字人", "机器人", "智能", "自动化", "Agent", "RAG", "AIGC", "智能体"],
    "智能体": ["Agent", "Copilot", "数字员工", "LangChain", "LlamaIndex", "AutoGPT", "Coze", "Dify", "扣子", "工作流", "Workflow", "编排", "RAG", "知识库", "向量", "Embedding", "工具调用", "Function Call", "多智能体", "Multi-Agent", "Swarm", "CrewAI", "AutoGen"],
    "大模型": ["OpenAI", "GPT", "文心", "通义", "Llama", "微调", "训练", "部署", "推理", "Token", "向量", "Prompt", "提示词"],
    "算力": ["GPU", "显卡", "英伟达", "H800", "4090", "服务器", "云计算", "智算中心", "租赁", "托管"],
    "芯片": ["半导体", "集成电路", "英伟达", "华为昇腾", "寒武纪", "FPGA", "ASIC"],
    "数据": ["标注", "清洗", "采集", "语料", "数据集", "版权", "向量数据库"],
    "数字人": ["直播", "短视频", "IP", "形象", "克隆", "配音", "虚拟人", "元宇宙"],
    "具身智能": ["机器人", "机械臂", "无人机", "自动驾驶", "传感器", "视觉", "雷达", "端侧模型"],
    "AIGC": ["生成式AI", "文本生成", "图像生成", "视频生成", "音乐生成", "代码生成"],
    "AI短剧": ["短剧", "视频", "内容创作", "剪辑", "拍摄", "流量", "完播率", "点赞", "评论", "转发"]
}

def compute_expert_score(text_a: str, text_b: str) -> float:
    """
    计算两个文本的匹配度，结合了字符相似度和专家规则语义匹配。
    """
    if not text_a or not text_b:
        return 0.0
    
    # 1. 基础字符相似度 (Base Character Similarity)
    # difflib 计算最长公共子序列，处理 "软件开发" vs "软件工程" 这种字面相似
    base_score = difflib.SequenceMatcher(None, text_a, text_b).ratio()
    
    # 2. 语义增强 (Semantic Boost)
    # 通过同义词库建立 "上市" <-> "证券" 这种非字面联系
    semantic_boost = 0.0
    
    # 归一化处理
    str_a = str(text_a).strip()
    str_b = str(text_b).strip()
    
    found_match = False
    
    # 检查 A 中的关键词是否匹配 B 中的关联词
    for key, related_words in BUSINESS_THESAURUS.items():
        if key in str_a:
            # 如果 A 包含 "上市"，检查 B 是否包含 ["证券", "投行"...]
            for word in related_words:
                if word in str_b:
                    semantic_boost = 0.6 # 给予显著加分
                    found_match = True
                    break
        if found_match: break
        
    # 双向检查：检查 B 中的关键词是否匹配 A 中的关联词
    if not found_match:
         for key, related_words in BUSINESS_THESAURUS.items():
            if key in str_b:
                for word in related_words:
                    if word in str_a:
                        semantic_boost = 0.6
                        found_match = True
                        break
            if found_match: break
            
    # 最终分数：基础分 + 语义分，上限 1.0
    # 这样 "尽快上市" (A) vs "证券行业" (B):
    # base_score ≈ 0
    # semantic_boost = 0.6 (因为 "上市" -> "证券")
    # total = 0.6 -> 属于高匹配
    return min(base_score + semantic_boost, 1.0)

def calculate_matches_task(user_industry: str, user_vision: str, others: List[Dict]) -> Dict:
    """
    CPU 密集型匹配任务，将在子进程中运行。
    """
    matches = {
        "customers": [], # My Industry vs Their Vision
        "partners": [],  # My Vision vs Their Industry
        "peers": []      # My Industry vs Their Industry
    }

    for other in others:
        # Handle potential None values safely
        other_ind_comp = other.get('industry_company') or ''
        other_bus_scope = other.get('business_scope') or ''
        other_industry = f"{other_ind_comp} {other_bus_scope}".strip()
        other_vision = other.get('vision_2026') or ""
        
        # 3.1 Customers (They need me)
        # My Industry (Supply) matches Their Vision (Demand)
        if user_industry and other_vision:
            score = compute_expert_score(user_industry, other_vision)
            if score > 0.15:
                matches["customers"].append({**other, "score": score})

        # 3.2 Partners (I need them)
        # My Vision (Demand) matches Their Industry (Supply)
        if user_vision and other_industry:
            score = compute_expert_score(user_vision, other_industry)
            if score > 0.15:
                matches["partners"].append({**other, "score": score})

        # 3.3 Peers (Same industry)
        # My Industry matches Their Industry
        if user_industry and other_industry:
            score = compute_expert_score(user_industry, other_industry)
            if score > 0.2:
                matches["peers"].append({**other, "score": score})

    # 4. Sort and Limit
    for key in matches:
        matches[key].sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top 5 per category
        matches[key] = matches[key][:5]
        
        # Hide sensitive info
        for p in matches[key]:
            # Safe phone masking
            p_phone = p.get('phone', '')
            if len(p_phone) >= 7:
                p['phone'] = p_phone[:3] + "****" + p_phone[-4:]
            else:
                p['phone'] = "****"
            
            p['location'] = "???" # Hidden location
            p['unlocked'] = False
            
    return matches


# Initialize Connection Pool
postgreSQL_pool = None

def init_db_pool():
    global postgreSQL_pool
    if postgreSQL_pool:
        try:
            postgreSQL_pool.closeall()
        except:
            pass
            
    db_config = {
        "host": CONFIG.get("db_host", "localhost"),
        "port": CONFIG.get("db_port", "5432"),
        "user": CONFIG.get("db_user", "gsdh"),
        "password": CONFIG.get("db_password", "123gsdh"),
        "database": CONFIG.get("db_name", "gsdh")
    }
    
    try:
        postgreSQL_pool = psycopg2.pool.ThreadedConnectionPool(1, 20, **db_config)
        print(f"PostgreSQL connection pool created successfully for {db_config['host']}:{db_config['port']}")
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while connecting to PostgreSQL", error)

init_db_pool()

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
    max_retries = 5
    for attempt in range(max_retries):
        conn = None
        try:
            conn = postgreSQL_pool.getconn()
            if conn.closed:
                # Should not happen with getconn() usually, but just in case
                postgreSQL_pool.putconn(conn, close=True)
                continue
                
            try:
                with conn.cursor() as cur:
                    cur.execute('SELECT 1')
                return conn
            except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.DatabaseError):
                # Connection is dead, remove it from pool
                if conn:
                    postgreSQL_pool.putconn(conn, close=True)
                # Loop will continue to get next connection
                continue
        except Exception as e:
            if conn:
                postgreSQL_pool.putconn(conn, close=True)
            print(f"Error getting DB connection (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise e
            
    raise Exception("Failed to get a valid database connection after retries")

def release_db_connection(conn):
    if conn:
        postgreSQL_pool.putconn(conn)

def assign_seat(cur, user_industry: str) -> str:
    """
    Allocate a seat based on:
    1. Even distribution (11 tables, max 12 per table)
       - Prioritize filling empty tables first (min count)
       - If counts are equal, fill sequentially (Table 1 before Table 2)
    2. Mix industries (try to put user in a table where their industry is least represented)
       - Uses natural language similarity to judge industry overlap
    """
    # Check if seating is enabled
    if not CONFIG.get("enable_seating", True):
        return "自由席"

    TOTAL_TABLES = int(CONFIG.get("total_tables", 14))
    MAX_PER_TABLE = int(CONFIG.get("max_per_table", 10))
    
    # Initialize table stats
    tables = {i: {'count': 0, 'industries': []} for i in range(1, TOTAL_TABLES + 1)}
    
    # Fetch current seating status
    # Uses aggregation for efficiency as requested
    # We use array_agg to collect industries for the diversity check in one query
    # Update: Include business_scope from checkin_info for more detailed matching
    # We concatenate industry_company (from gsdh_data) and business_scope (from checkin_info)
    query = """
    SELECT 
        ci.location, 
        COUNT(ci.gsdh_id), 
        array_agg(
            COALESCE(gd.industry_company, '') || ' ' || COALESCE(ci.business_scope, '')
        )
    FROM checkin_info ci 
    LEFT JOIN gsdh_data gd ON ci.gsdh_id = gd.new_id 
    WHERE ci.location IS NOT NULL AND ci.location LIKE '第%桌'
    GROUP BY ci.location
    """
    cur.execute(query)
    rows = cur.fetchall()
    
    for row in rows:
        loc = row[0]
        count = row[1]
        industries = row[2] if row[2] else []
        
        try:
            # Extract table number
            table_num = int(loc.replace("第", "").replace("桌", ""))
            if 1 <= table_num <= TOTAL_TABLES:
                tables[table_num]['count'] = count
                # Filter out None values from industries
                tables[table_num]['industries'] = [ind for ind in industries if ind]
        except ValueError:
            continue
            
    # Filter tables that are not full
    available_tables = [t for t in tables.items() if t[1]['count'] < MAX_PER_TABLE]
    
    if not available_tables:
        return "自由席" # Fallback if all full
        
    # Strategy 1: Find tables with Minimum Count
    # This automatically handles "Prioritize filling TOTAL_TABLES" because empty tables have count 0
    min_count = min(t[1]['count'] for t in available_tables)
    candidates = [t for t in available_tables if t[1]['count'] == min_count]
    
    # Sort by table number to ensure sequential filling if counts are equal (Requirement: 顺序分配)
    candidates.sort(key=lambda x: x[0])
    
    # Strategy 2: Optimize for Industry Diversity
    if not user_industry:
        # If no industry info, just pick the first one (Sequential)
        best_table = candidates[0][0]
    else:
        # Calculate similarity scores
        # Score = sum of similarity with existing users
        # Lower score is better (more unique)
        scored_candidates = []
        
        for table_id, stats in candidates:
            total_similarity = 0.0
            for existing_ind in stats['industries']:
                if existing_ind:
                    # Use Expert Score for better semantic matching
                    # This helps understand "Natural Language" industries better than exact match
                    sim = compute_expert_score(user_industry, existing_ind)
                    total_similarity += sim
            
            scored_candidates.append((table_id, total_similarity))
        
        # Sort by similarity score (asc), then by table_id (asc)
        scored_candidates.sort(key=lambda x: (x[1], x[0]))
        
        best_table = scored_candidates[0][0]
        
    return f"第{best_table}桌"

def get_tablemates(cur, location: str, exclude_id: str, user_vision: str = "", user_industry: str = "") -> List[Dict]:
    """
    Get 3 tablemates based on:
    1. Vision similarity: Match tablemate's vision_2026 with user's vision_2026 (Find similar goals)
    2. Supply-Demand match: Match tablemate's vision_2026 with user's industry (Find potential partners)
    """
    if not location or location == "自由席":
        return []
    
    # Debug: Check who is at this location
    print(f"DEBUG: Fetching tablemates for location: '{location}', excluding: '{exclude_id}'")
    
    # Updated query to fetch more details from checkin_info
    query = """
    SELECT ci.name, gd.industry_company, ci.company_name, ci.position, ci.business_scope, ci.vision_2026
    FROM checkin_info ci
    LEFT JOIN gsdh_data gd ON ci.gsdh_id = gd.new_id
    WHERE ci.location = %s AND ci.gsdh_id != %s
    """
    cur.execute(query, (location, exclude_id))
    rows = cur.fetchall()
    
    print(f"DEBUG: Found {len(rows)} potential tablemates")
    
    candidates = []
    for row in rows:
        candidate = {
            "name": row[0],
            "industry": row[1] or "暂无行业信息",
            "company_name": row[2] or "暂无单位信息",
            "position": row[3] or "暂无职务信息",
            "business_scope": row[4] or "暂无业务信息",
            "vision_2026": row[5] or "",
            "match_type": [],
            "score": 0.0
        }
        candidates.append(candidate)

    if not candidates:
        return []

    # Scoring Logic
    for cand in candidates:
        cand_vision = cand["vision_2026"]
        cand_industry = cand["industry"]
        
        # 1. Vision Similarity (Find peers with similar goals)
        if user_vision and cand_vision:
            sim = compute_expert_score(user_vision, cand_vision)
            # Weight this score
            cand["score"] += sim * 1.0
            if sim > 0.25: # Threshold for "similarity" (Adjusted to 0.25 for better precision)
                cand["match_type"].append("志同道合 (愿景相似)")

        # 2. Cross Match: My Industry matches Their Vision (I can help them)
        if user_industry and cand_vision:
             sim = compute_expert_score(user_industry, cand_vision)
             cand["score"] += sim * 1.5 # Give higher weight to potential business match
             if sim > 0.25:
                 cand["match_type"].append("潜在合作 (您的行业匹配对方愿景)")
        
        # 3. Cross Match: Their Industry matches My Vision (They can help me)
        if user_vision and cand_industry:
             sim = compute_expert_score(user_vision, cand_industry)
             cand["score"] += sim * 1.5
             if sim > 0.25:
                 cand["match_type"].append("潜在贵人 (对方行业匹配您的愿景)")

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top 3
    top_candidates = candidates[:3]
    
    # Format output
    result = []
    for cand in top_candidates:
        # If no specific match type, just say "同桌伙伴"
        match_reason = " | ".join(cand["match_type"]) if cand["match_type"] else "同桌伙伴"
        
        result.append({
            "name": cand["name"],
            "industry": cand["industry"],
            "company_name": cand["company_name"],
            "position": cand["position"],
            "business_scope": cand["business_scope"],
            "vision_2026": cand["vision_2026"] or "暂无愿景信息",
            "match_reason": match_reason
        })
        
    return result

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    global CONFIG
    CONFIG = load_config() # Reload config on refresh
    # Pass config as json string for JS usage if needed, or individual fields
    return templates.TemplateResponse("index.html", {"request": request, "config": CONFIG})

class UnlockRequest(BaseModel):
    my_phone: str
    target_id: str

class ResourceMatchRequest(BaseModel):
    phone: str

@app.get("/search", response_class=HTMLResponse)
async def resource_match_page(request: Request):
    return templates.TemplateResponse("resource_match.html", {"request": request})

@app.post("/api/resource-match")
def resource_match(req: ResourceMatchRequest):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 1. Fetch current user
        # Optimize: Only fetch necessary fields
        cur.execute("""
            SELECT gd.new_id, gd.name, gd.phone, ci.social_point as points, 
                   gd.industry_company, ci.business_scope, ci.vision_2026
            FROM gsdh_data gd
            LEFT JOIN checkin_info ci ON gd.new_id = ci.gsdh_id
            WHERE gd.phone = %s
        """, (req.phone,))
        user = cur.fetchone()

        if not user:
            cur.close()
            release_db_connection(conn)
            return JSONResponse(content={"success": False, "message": "用户不存在"}, status_code=404)

        user_industry = f"{user['industry_company'] or ''} {user['business_scope'] or ''}".strip()
        user_vision = user['vision_2026'] or ""

        # 2. Fetch ALL other users (who have checked in)
        # Performance Note: Fetching all rows is slow if N is large.
        # But for N < 1000 it's acceptable. For larger N, we need vector search (e.g. pgvector).
        # We limit the fields to reduce payload size.
        cur.execute("""
            SELECT gd.new_id, gd.name, gd.phone, gd.industry_company, 
                   ci.company_name, ci.position, ci.business_scope, ci.vision_2026, ci.location
            FROM checkin_info ci
            JOIN gsdh_data gd ON ci.gsdh_id = gd.new_id
            WHERE gd.new_id != %s
        """, (user['new_id'],))
        others = cur.fetchall()

        cur.close()
        release_db_connection(conn)

        # 3. Calculate Matches (Using Process Pool for Concurrency)
        # 将 CPU 密集型计算提交给进程池，避免阻塞主进程和 GIL 锁竞争
        if process_pool:
            future = process_pool.submit(calculate_matches_task, user_industry, user_vision, others)
            matches = future.result() # Wait for result (blocks this thread, but not the whole server)
        else:
            # Fallback if pool not initialized
            matches = calculate_matches_task(user_industry, user_vision, others)

        return {
            "success": True, 
            "user": {
                "name": user['name'],
                "industry_company": user['industry_company'],
                "points": user['points'] if user['points'] is not None else 0,
                "phone": user['phone']
            },
            "matches": matches
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'conn' in locals() and conn:
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.post("/api/unlock-contact")
def unlock_contact(req: UnlockRequest):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Check User Points
        cur.execute("""
            SELECT ci.social_point as points 
            FROM checkin_info ci
            JOIN gsdh_data gd ON ci.gsdh_id = gd.new_id
            WHERE gd.phone = %s
        """, (req.my_phone,))
        res = cur.fetchone()
        
        if not res:
            cur.close()
            release_db_connection(conn)
            return JSONResponse(content={"success": False, "message": "用户未签到或不存在"}, status_code=404)
        
        points = res['points'] if res['points'] is not None else 0
        if points <= 0:
            cur.close()
            release_db_connection(conn)
            return JSONResponse(content={"success": False, "message": "积分不足"}, status_code=400)
            
        # 2. Deduct Point
        # Update checkin_info using a subquery to map phone to gsdh_id
        cur.execute("""
            UPDATE checkin_info 
            SET social_point = social_point - 1 
            WHERE gsdh_id = (SELECT new_id FROM gsdh_data WHERE phone = %s)
        """, (req.my_phone,))
        
        # 3. Fetch Target Info
        cur.execute("""
            SELECT gd.phone, ci.location 
            FROM gsdh_data gd 
            LEFT JOIN checkin_info ci ON gd.new_id = ci.gsdh_id
            WHERE gd.new_id = %s
        """, (req.target_id,))
        target = cur.fetchone()
        
        conn.commit()
        cur.close()
        release_db_connection(conn)
        
        return {
            "success": True,
            "remaining_points": points - 1,
            "contact": {
                "phone": target['phone'],
                "location": target['location'] or "未分配座位"
            }
        }

    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.get("/api/search")
def search_user(query: str):
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
                cur.close()
                release_db_connection(conn)
                
                # Check if ticket validation is disabled
                if not CONFIG.get('enable_ticket_validation', True):
                    # If validation is disabled, allow creating a new user
                    return JSONResponse(content={
                        "found": False, 
                        "allow_create": True, 
                        "user": {"name": query if not query.isdigit() else "", "phone": query if query.isdigit() else ""}
                    })

                return JSONResponse(content={"found": False, "message": "未查询到相关信息，请检查输入是否正确"}, status_code=404)
            elif len(users) > 1:
                # If multiple users found by name, return list for user to select (simplified here to return first or error)
                # For this MVP, let's return all matching users so frontend can handle selection
                cur.close()
                release_db_connection(conn)
                return JSONResponse(content={"found": True, "multiple": True, "users": users})
            else:
                user = users[0]
        
        # Check if already signed
        if user.get('is_signed') == 'TRUE':
             # Check if already signed
             cur.execute("SELECT location, vision_2026 FROM checkin_info WHERE gsdh_id = %s", (user['new_id'],))
             checkin_info = cur.fetchone() # Fetch as RealDictRow
             
             assigned_seat = checkin_info['location'] if checkin_info else "自由席"
             
             # Fetch tablemates
             # Use a new cursor for the helper function to avoid cursor factory conflict or state issues
             # Note: get_tablemates expects a standard cursor for tuple results, but here we have DictCursor
             # We can adapt get_tablemates or just use key access if we pass the DictCursor
             # Let's create a fresh standard cursor to be safe and consistent with get_tablemates implementation
             cur_plain = conn.cursor() 
             
             # Fetch user's vision and industry for matching
             user_vision = checkin_info.get('vision_2026', '') if checkin_info else ''
             # user['industry_company'] is already available in user dict
             user_industry = user.get('industry_company', '')

             tablemates = get_tablemates(cur_plain, assigned_seat, user['new_id'], user_vision, user_industry)
             cur_plain.close()
             
             cur.close()
             release_db_connection(conn)
             return JSONResponse(content={
                 "found": True, 
                 "user": user, 
                 "already_signed": True, 
                 "seat": assigned_seat,
                 "tablemates": tablemates
             })
        
        cur.close()
        release_db_connection(conn)
        return JSONResponse(content={"found": True, "user": user, "already_signed": False})

    except Exception as e:
        if 'conn' in locals() and conn:
            release_db_connection(conn)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/checkin")
def checkin_user(checkin_data: CheckinRequest):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        final_gsdh_id = checkin_data.gsdh_id
        
        # === Handle New User (TEMP ID) ===
        if checkin_data.gsdh_id.startswith("TEMP_"):
            # 1. Calculate new real ID (Max + 1)
            # Find max numeric new_id. Using regex to ensure we only look at numbers.
            cur.execute("SELECT MAX(CAST(new_id AS INTEGER)) FROM gsdh_data WHERE new_id ~ '^[0-9]+$'")
            row = cur.fetchone()
            max_id = row[0] if row and row[0] is not None else 0
            new_real_id = str(max_id + 1)
            
            # 2. Insert into gsdh_data first to satisfy Foreign Key
            insert_user_sql = """
            INSERT INTO gsdh_data (new_id, name, phone, industry_company, fee, payment_channel, is_signed)
            VALUES (%s, %s, %s, %s, '0', 'onsite_checkin', 'TRUE')
            """
            # Use provided company name or default
            industry_val = checkin_data.company_name or "现场注册"
            
            cur.execute(insert_user_sql, (
                new_real_id,
                checkin_data.name,
                checkin_data.phone,
                industry_val
            ))
            
            # Update ID for subsequent operations
            final_gsdh_id = new_real_id
        
        # 0. Get user's industry from gsdh_data to help with seat allocation
        base_industry = ""
        enable_validation = CONFIG.get('enable_ticket_validation', True)

        if enable_validation:
            cur.execute("SELECT industry_company FROM gsdh_data WHERE new_id = %s", (final_gsdh_id,))
            res = cur.fetchone()
            base_industry = res[0] if res and res[0] else ""
        else:
            # If validation is disabled and NOT a temp user (already handled above), ensure user exists
            # (Though logic above handles TEMP users, existing users might still need upsert if data is inconsistent)
            if not checkin_data.gsdh_id.startswith("TEMP_"):
                 cur.execute("""
                     INSERT INTO gsdh_data (new_id, name, phone, is_signed, industry_company, fee, payment_channel)
                     VALUES (%s, %s, %s, 'TRUE', '现场签到', '0', 'skipped')
                     ON CONFLICT (new_id) DO UPDATE SET is_signed = 'TRUE'
                 """, (final_gsdh_id, checkin_data.name, checkin_data.phone))
        
        # Combine base industry with the newly provided business_scope for better matching
        user_industry_info = f"{base_industry} {checkin_data.business_scope or ''}".strip()
        
        # 1. Allocate Seat
        assigned_seat = assign_seat(cur, user_industry_info)
        
        # 2. Insert into checkin_info with assigned seat
        # Initialize social_point to 5
        insert_sql = """
        INSERT INTO checkin_info 
        (name, phone, company_name, position, business_scope, vision_2026, location, gsdh_id, social_point)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 4)
        """
        cur.execute(insert_sql, (
            checkin_data.name,
            checkin_data.phone,
            checkin_data.company_name,
            checkin_data.position,
            checkin_data.business_scope,
            checkin_data.vision_2026,
            assigned_seat,  # Use the generated seat
            final_gsdh_id   # Use the real ID
        ))
        
        # 3. Update gsdh_data is_signed to TRUE (if not already done)
        if not checkin_data.gsdh_id.startswith("TEMP_"):
            update_sql = "UPDATE gsdh_data SET is_signed = 'TRUE' WHERE new_id = %s"
            cur.execute(update_sql, (final_gsdh_id,))
        
        conn.commit()
        
        # 4. Fetch tablemates for the newly assigned seat
        # Use provided vision and industry for matching
        tablemates = get_tablemates(cur, assigned_seat, final_gsdh_id, checkin_data.vision_2026 or "", user_industry_info)
        
        cur.close()
        release_db_connection(conn)
        
        return {"success": True, "message": "签到成功！", "seat": assigned_seat, "tablemates": tablemates}
        
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": f"签到失败: {str(e)}"}, status_code=500)

@app.get("/add-user", response_class=HTMLResponse)
async def add_user_page(request: Request):
    global CONFIG
    CONFIG = load_config()
    secret = os.getenv("ADD_USER_SECRET", "123quant-speed")
    print(f"DEBUG: Secret loaded: '{secret}'")
    return templates.TemplateResponse("add_user.html", {"request": request, "secret": secret, "config": CONFIG})

class UpdateUserRequest(BaseModel):
    gsdh_id: str
    name: str
    phone: str
    company_name: Optional[str] = None
    position: Optional[str] = None
    business_scope: Optional[str] = None
    vision_2026: Optional[str] = None
    is_signed: Optional[str] = None
    location: Optional[str] = None
    fee: Optional[str] = None
    payment_channel: Optional[str] = None

class UncheckinRequest(BaseModel):
    """
    取消签到请求：
    - gsdh_id：用户唯一 ID
    - 可选覆盖基础信息字段：name、company_name
    """
    gsdh_id: str
    name: Optional[str] = None
    company_name: Optional[str] = None

@app.get("/edit", response_class=HTMLResponse)
async def edit_page(request: Request):
    global CONFIG
    CONFIG = load_config()
    return templates.TemplateResponse("edit.html", {"request": request, "config": CONFIG})

@app.get("/api/get-user-details")
def get_user_details(phone: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                gd.name as name_base, gd.industry_company as company_base, gd.is_signed, gd.fee, gd.payment_channel,
                ci.name as name_checkin, ci.company_name as company_checkin,
                ci.position, ci.business_scope, ci.vision_2026, ci.location
            FROM gsdh_data gd
            LEFT JOIN checkin_info ci ON gd.new_id = ci.gsdh_id
            WHERE gd.phone = %s
        """, (phone,))
        
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        
        if row:
            data = {
                "name": row['name_checkin'] or row['name_base'],
                "company_name": row['company_checkin'] or row['company_base'],
                "position": row['position'],
                "business_scope": row['business_scope'],
                "vision_2026": row['vision_2026'],
                "is_signed": row['is_signed'],
                "location": row['location'],
                "fee": row['fee'],
                "payment_channel": row['payment_channel']
            }
            return {"success": True, "data": data}
        else:
            return JSONResponse(content={"success": False, "message": "User not found"}, status_code=404)
            
    except Exception as e:
        if 'conn' in locals() and conn:
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.post("/api/update-user")
def update_user(req: UpdateUserRequest):
    """
    仅覆盖已有数据的编辑接口：
    - 总是更新基础表 gsdh_data
    - 仅当存在对应的 checkin_info 时覆盖更新该记录
    - 不会新增新的 checkin_info 记录（编辑不触发创建）
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Update gsdh_data (Base info)
        cur.execute("""
            UPDATE gsdh_data 
            SET name = %s, industry_company = %s, is_signed = %s, fee = %s, payment_channel = %s
            WHERE new_id = %s
        """, (req.name, req.company_name, req.is_signed, req.fee, req.payment_channel, req.gsdh_id))
        
        # 2. Check if checkin_info exists
        cur.execute("SELECT 1 FROM checkin_info WHERE gsdh_id = %s", (req.gsdh_id,))
        exists = cur.fetchone()
        
        if exists:
            # Update existing checkin_info
            cur.execute("""
                UPDATE checkin_info
                SET name = %s, company_name = %s, position = %s, business_scope = %s, vision_2026 = %s, location = %s
                WHERE gsdh_id = %s
            """, (req.name, req.company_name, req.position, req.business_scope, req.vision_2026, req.location, req.gsdh_id))
            msg = "更新成功"
        else:
            # Do not create new checkin_info on edit, but return success for base info update
            msg = "基础信息已更新（未签到用户暂存部分详情）"
        
        conn.commit()
        cur.close()
        release_db_connection(conn)
        
        return {"success": True, "message": msg}
        
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.post("/api/uncheckin")
def uncheckin_user(req: UncheckinRequest):
    """
    取消签到并覆盖基础信息：
    - 删除该用户的 checkin_info 记录
    - 将 gsdh_data.is_signed 置为 FALSE
    - 可选同时更新 gsdh_data 的 name、industry_company
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 删除签到记录
        cur.execute("DELETE FROM checkin_info WHERE gsdh_id = %s", (req.gsdh_id,))
        
        # 更新基础信息与签到状态
        if req.name is not None and req.company_name is not None:
            cur.execute("""
                UPDATE gsdh_data
                SET name = %s, industry_company = %s, is_signed = 'FALSE'
                WHERE new_id = %s
            """, (req.name, req.company_name, req.gsdh_id))
        elif req.name is not None:
            cur.execute("""
                UPDATE gsdh_data
                SET name = %s, is_signed = 'FALSE'
                WHERE new_id = %s
            """, (req.name, req.gsdh_id))
        elif req.company_name is not None:
            cur.execute("""
                UPDATE gsdh_data
                SET industry_company = %s, is_signed = 'FALSE'
                WHERE new_id = %s
            """, (req.company_name, req.gsdh_id))
        else:
            cur.execute("UPDATE gsdh_data SET is_signed = 'FALSE' WHERE new_id = %s", (req.gsdh_id,))
        
        conn.commit()
        cur.close()
        release_db_connection(conn)
        return {"success": True, "message": "已取消签到并更新基础信息"}
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": f"取消签到失败: {str(e)}"}, status_code=500)

@app.post("/api/add-user")
def add_user_api(user_data: AddUserRequest):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if phone already exists
        cur.execute("SELECT * FROM gsdh_data WHERE phone = %s", (user_data.phone,))
        if cur.fetchone():
            cur.close()
            release_db_connection(conn)
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
        release_db_connection(conn)
        
        return {"success": True, "message": "添加成功", "new_id": new_id}
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": f"添加失败: {str(e)}"}, status_code=500)

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """
    渲染管理后台页面。
    """
    secret = os.getenv("ADD_USER_SECRET", "123quant-speed")
    return templates.TemplateResponse("admin.html", {"request": request, "secret": secret})

@app.get("/wall", response_class=HTMLResponse)
async def wall_page(request: Request):
    """
    渲染签到大屏页面。
    """
    global CONFIG
    CONFIG = load_config()
    return templates.TemplateResponse("wall.html", {"request": request, "config": CONFIG})

@app.get("/api/wall/data")
def get_wall_data():
    """
    获取大屏所需的数据：
    1. 所有已签到用户的 Company Name, Vision, Business Scope
    2. 过滤掉空数据
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # 获取最新的签到数据（按时间倒序）
        query = """
        SELECT gsdh_id, name, company_name, position, business_scope, vision_2026, social_point
        FROM checkin_info
        ORDER BY created_at DESC
        """
        cur.execute(query)
        rows = cur.fetchall()
        
        cur.close()
        release_db_connection(conn)
        
        return {"success": True, "data": rows}
    except Exception as e:
        if 'conn' in locals() and conn:
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.get("/api/admin/config")
def get_config():
    """
    获取当前的配置信息。
    """
    return load_config()

@app.post("/api/admin/config")
def update_config(config: Dict):
    """
    更新配置信息并保存到文件。
    """
    global CONFIG
    save_config(config)
    CONFIG = config
    # Re-initialize DB pool with new config
    init_db_pool()
    return {"success": True}

@app.post("/api/admin/test-db")
def test_db_connection(config: Dict):
    """
    测试数据库连接
    """
    try:
        conn = psycopg2.connect(
            host=config.get("db_host"),
            port=config.get("db_port"),
            user=config.get("db_user"),
            password=config.get("db_password"),
            database=config.get("db_name"),
            connect_timeout=5
        )
        conn.close()
        return {"success": True, "message": "连接成功"}
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=400)

@app.post("/api/admin/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片文件到 static 目录。
    """
    try:
        file_location = f"static/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return {"success": True, "url": f"/static/{file.filename}"}
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.post("/api/admin/reset-db")
def reset_database():
    """
    重置数据库：删除并重新创建 gsdh_data 和 checkin_info 表。
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Drop existing tables
        cur.execute("DROP TABLE IF EXISTS checkin_info CASCADE;")
        cur.execute("DROP TABLE IF EXISTS gsdh_data CASCADE;")
        
        # Recreate tables
        # gsdh_data
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gsdh_data (
                new_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100),
                phone VARCHAR(20) UNIQUE,
                industry_company VARCHAR(200),
                fee VARCHAR(50),
                payment_channel VARCHAR(50),
                is_signed VARCHAR(10) DEFAULT 'FALSE'
            );
        """)
        
        # checkin_info
        cur.execute("""
            CREATE TABLE IF NOT EXISTS checkin_info (
                id SERIAL PRIMARY KEY,
                gsdh_id VARCHAR(50) REFERENCES gsdh_data(new_id),
                name VARCHAR(100),
                phone VARCHAR(20),
                company_name VARCHAR(200),
                position VARCHAR(100),
                business_scope TEXT,
                vision_2026 TEXT,
                location VARCHAR(50),
                social_point INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Initial Data (Optional - add a test user)
        cur.execute("""
            INSERT INTO gsdh_data (new_id, name, phone, industry_company, fee, payment_channel, is_signed)
            VALUES ('1', '测试用户', '13800000000', '科技', '0', 'test', 'FALSE');
        """)
        
        conn.commit()
        cur.close()
        release_db_connection(conn)
        return {"success": True, "message": "数据库已重置"}
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
            release_db_connection(conn)
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description='Run the Checkin System.')
    parser.add_argument('--port', type=int, default=8800, help='Port to run the server on')
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
