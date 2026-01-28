I have analyzed the current project structure and designed a "Check-in Large Screen" (签到大屏) solution that integrates seamlessly with your existing system.

### **Plan Overview**

I will implement a **"Galaxy Spotlight"** style large screen:

1. **Background (Danmaku/Floating Tags)**:

   * `Business Scope` and short `Vision` keywords will float elegantly in the background like stars or drifting clouds.

   * Uses the project's blue/cyan tech theme.

2. **Foreground (Spotlight Card)**:

   * A central, animated card that cycles through checked-in guests.

   * Displays: **Company Name** (Prominent), **Guest Name/Position**, and **Vision 2026**.

   * Updates every 8-10 seconds to ensure everyone gets exposure.

### **Implementation Steps**

#### 1. Backend (`main.py`)

* Add a new route `/wall` to serve the large screen page.

* Add an API `/api/wall/data` to fetch approved check-in data (filtering out empty entries).

#### 2. Frontend (`templates/wall.html`)

* Create a new responsive HTML template.

* **Visuals**: Reusing the `radial-gradient` background and neon aesthetics from `index.html`.

* **Animation**:

  * JS-based floating animation for background tags.

  * CSS transitions for the central spotlight card.

#### 3. Admin Integration (`templates/admin.html`)

* Add a button "Open Large Screen" (打开签到大屏) in the Admin Dashboard for easy access.

### **Technical Details**

* **Tech Stack**: Native JS/CSS (no extra heavy libraries needed), integrated into the existing FastAPI app.

* **Data Source**: Reads directly from `checkin_info` table.

* **Compatibility**: Optimized for 1080p/4K displays (MacOS standard).

