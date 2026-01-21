# Add Seating Arrangement Settings & Style Update Plan

## 1. Backend Updates (`main.py`)
- **Update `DEFAULT_CONFIG`**: Add `enable_seating` (bool, default True), `total_tables` (int, default 14), and `max_per_table` (int, default 10).
- **Update `assign_seat` function**: 
    - Use `CONFIG["total_tables"]` and `CONFIG["max_per_table"]` instead of hardcoded values.
    - Check `CONFIG["enable_seating"]`. If False, return "自由席" immediately.

## 2. Admin Page Updates (`templates/admin.html`)
- **Style Overhaul**: Replace the current light theme CSS with the dark, futuristic theme from `index.html` to ensure consistency.
- **Add Seating Settings Section**:
    - Add a checkbox for "开启分桌功能".
    - Add input fields for "总桌数" and "每桌最大人数" (conditionally displayed when seating is enabled).
- **Update JS Logic**:
    - Load and save the new seating configuration fields.
    - Add toggle logic to show/hide table count inputs based on the checkbox.

## 3. Frontend Updates (`templates/index.html`)
- **Inject Config**: Ensure `enable_seating` config is accessible in JavaScript.
- **Conditional Display**:
    - In `submitCheckin` (new checkin) and `searchUser` (already signed), check `enable_seating`.
    - If disabled:
        - Hide seat number displays (`#seat-display`, `#signed-seat-display`).
        - Hide tablemate recommendations (`#tablemates-container`, `#signed-tablemates-container`).
        - Update text to indicate "自由入座" (Free Seating).
