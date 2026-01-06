# 云南AI共生大会签到系统

## 项目概述

这是一个专为云南AI共生大会设计的智能签到系统，基于FastAPI框架开发，提供完整的会议签到解决方案。系统集成了智能座位分配、嘉宾信息管理、同桌推荐等功能，为参会者提供流畅的签到体验。

## 主要功能

### 🎯 核心功能
- **智能签到流程**：支持手机号或姓名搜索，快速定位参会嘉宾
- **智能座位分配**：基于行业分布和人数均衡的算法自动分配座位
- **同桌嘉宾推荐**：签到后展示同table的3位随机嘉宾，促进交流
- **嘉宾信息补全**：签到时可补充单位、职务、业务范围等信息
- **重复签到检测**：自动识别已签到用户，避免重复操作

### 🔧 管理功能
- **新嘉宾添加**：支持现场添加新参会嘉宾
- **多用户搜索**：模糊搜索支持，可处理重名情况
- **信息验证**：手机号唯一性验证，防止重复注册

### 🎨 用户体验
- **现代化UI设计**：采用AI主题的未来主义风格界面
- **响应式布局**：适配各种设备屏幕
- **实时反馈**：清晰的操作提示和错误处理
- **模态框详情**：点击同桌嘉宾可查看详细信息

## 技术架构

### 后端技术栈
- **FastAPI**：高性能Python Web框架
- **PostgreSQL**：数据持久化存储
- **psycopg2**：PostgreSQL数据库适配器
- **Pydantic**：数据验证和序列化
- **Jinja2**：模板引擎
- **UUID**：唯一标识符生成

### 前端技术
- **HTML5 + CSS3**：现代化界面设计
- **原生JavaScript**：轻量级交互逻辑
- **AJAX**：异步数据交互
- **响应式设计**：移动端友好

### 部署方案
- **Docker容器化**：便于部署和扩展
- **自动化脚本**：一键构建和部署
- **多架构支持**：支持AMD64和ARM64架构

## 项目结构

```
.
├── main.py                 # 主应用文件，包含所有API端点
├── templates/              # HTML模板目录
│   ├── index.html         # 主签到页面
│   └── add_user.html      # 添加新嘉宾页面
├── static/                 # 静态资源目录
│   └── image.jpg          # 会议主题图片
├── docker_deply.sh        # Docker部署自动化脚本
├── gsdh.csv              # 嘉宾数据文件（示例）
└── README.md             # 项目文档
```

## 核心算法

### 座位分配算法
系统采用双层策略进行智能座位分配：

1. **均衡分布**：优先选择人数最少的table，确保13个table人数均衡（每桌最多12人）
2. **行业多样性**：在人数最少的table中，选择该行业占比最少的table，促进行业交流

```python
def assign_seat(cur, user_industry: str) -> str:
    # 1. 统计现有table人数和行业分布
    # 2. 筛选未满的table（每桌≤12人）
    # 3. 选择人数最少的table
    # 4. 在候选table中选择该行业占比最少的
    # 5. 返回分配的桌号（如"第5桌"）
```

### 同桌推荐算法
- 随机选择同table的3位嘉宾（排除自己）
- 展示基本信息：姓名、行业、公司、职务、业务范围、2026愿景
- 支持点击查看详细信息

## API接口文档

### 搜索接口
```http
GET /api/search?query={phone_or_name}
```
- 支持手机号精确搜索和姓名模糊搜索
- 返回用户信息和签到状态

### 签到接口
```http
POST /api/checkin
Content-Type: application/json

{
    "gsdh_id": "uuid",
    "name": "姓名",
    "phone": "手机号",
    "company_name": "公司名称",
    "position": "职务",
    "business_scope": "业务范围",
    "vision_2026": "2026愿景",
    "location": "座位号"
}
```

### 添加用户接口
```http
POST /api/add-user
Content-Type: application/json

{
    "name": "姓名",
    "phone": "手机号",
    "industry_company": "公司/行业"
}
```

## 数据库设计

### 主要数据表

#### gsdh_data（嘉宾基础信息表）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| new_id | UUID | 主键，唯一标识 |
| name | VARCHAR | 姓名 |
| phone | VARCHAR | 手机号（唯一） |
| industry_company | VARCHAR | 公司/行业信息 |
| is_signed | BOOLEAN | 签到状态 |

#### checkin_info（签到信息表）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | SERIAL | 主键，自增 |
| gsdh_id | UUID | 关联gsdh_data |
| name | VARCHAR | 签到时确认的姓名 |
| phone | VARCHAR | 签到时确认的手机号 |
| company_name | VARCHAR | 公司名称 |
| position | VARCHAR | 职务 |
| business_scope | TEXT | 业务范围 |
| vision_2026 | TEXT | 2026年业务愿景 |
| location | VARCHAR | 分配的座位号 |
| created_at | TIMESTAMP | 签到时间 |

## 快速开始

### 环境要求
- Python 3.8+
- PostgreSQL 12+
- Docker（可选）

### 本地开发

1. **克隆项目**
```bash
git clone <repository-url>
cd yunnan-ai-conference-checkin
```

2. **安装依赖**
```bash
pip install fastapi uvicorn psycopg2-binary pydantic jinja2
```

3. **配置数据库**
```bash
# 创建数据库
createdb gsdh

# 执行SQL初始化脚本（需要创建相关表结构）
```

4. **启动应用**
```bash
python main.py
# 或使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

5. **访问应用**
- 主签到页面：http://localhost:8000
- 添加嘉宾页面：http://localhost:8000/add-user

### Docker部署

1. **构建镜像**
```bash
# 完整构建和部署流程（默认AMD64架构）
./docker_deply.sh

# 构建ARM64架构镜像
./docker_deply.sh -arm

# 仅上传已存在的tar文件
./docker_deply.sh -upload
```

2. **配置参数**
编辑 `docker_deply.sh` 脚本顶部的配置：
```bash
SERVER_HOST="your-server-ip"      # 服务器IP地址
SERVER_USER="ubuntu"              # 服务器用户名
SERVER_PASSWORD="your-password"   # 服务器密码
SERVER_PORT="22"                  # SSH端口
```

3. **访问应用**
部署完成后，通过 `http://your-server-ip:8011` 访问

## 配置说明

### 数据库配置
在 `main.py` 中修改数据库连接参数：
```python
DB_CONFIG = {
    "host": "121.43.104.161",
    "port": "6432",
    "user": "gsdh",
    "password": "123gsdh",
    "database": "gsdh"
}
```

### 座位分配规则
- 总table数：13桌
- 每桌最大人数：12人
- 分配策略：人数均衡 + 行业多样性

## 使用流程

### 正常签到流程
1. 嘉宾输入手机号或姓名进行搜索
2. 系统显示匹配的嘉宾信息
3. 嘉宾确认并补充完整信息
4. 系统智能分配座位并展示同桌嘉宾
5. 签到完成，显示座位号

### 现场添加嘉宾
1. 访问添加嘉宾页面
2. 输入新嘉宾的基本信息
3. 系统验证手机号唯一性
4. 添加成功后，新嘉宾可进行正常签到

## 安全特性

- **数据验证**：所有输入都经过Pydantic严格验证
- **错误处理**：完善的异常捕获和用户友好的错误提示
- **手机号保护**：手机号信息不公开显示
- **防重复签到**：基于数据库事务确保数据一致性

## 性能优化

- **数据库连接池**：复用数据库连接，减少连接开销
- **索引优化**：在搜索字段上建立适当索引
- **缓存策略**：静态资源缓存，提升加载速度
- **异步处理**：非阻塞的API响应

## 扩展功能

### 未来可考虑的功能
- **二维码签到**：生成个人二维码，扫码快速签到
- **实时统计**：后台实时查看签到进度和统计
- **消息推送**：签到成功后发送确认短信/邮件
- **数据导出**：支持签到数据导出为Excel等格式
- **多语言支持**：支持中英文切换
- **人脸识别**：集成人脸识别技术，提升签到体验

## 故障排除

### 常见问题

**Q: 数据库连接失败**
A: 检查数据库配置是否正确，确保PostgreSQL服务正常运行

**Q: 座位分配算法异常**
A: 检查数据库中table数据是否完整，确保location字段格式正确

**Q: 搜索功能无结果**
A: 确认数据库中有对应的数据，检查搜索关键词是否正确

**Q: Docker部署失败**
A: 检查服务器配置和网络连接，确保Docker服务正常运行

## 维护建议

1. **定期备份**：定期备份数据库，防止数据丢失
2. **监控性能**：监控API响应时间和数据库性能
3. **日志分析**：定期分析应用日志，发现潜在问题
4. **安全更新**：及时更新依赖包，修复安全漏洞

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请联系项目维护者。

---

**特别说明**：本项目专为云南AI共生大会定制开发，结合了会议的实际需求和AI技术的特色，旨在为参会嘉宾提供便捷、智能的签到体验。