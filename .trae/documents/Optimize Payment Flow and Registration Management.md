I will optimize the registration and payment flow with the specific requirement to record the actual payment amount.

### 1. Backend Updates (`main.py`)
- **Database Schema**: Add `out_trade_no` column to `gsdh_data` to link users with payment orders.
- **Pre-payment Validation**:
  - Update `create_payment` (H5) and `create_native_payment`.
  - Check if phone exists in `gsdh_data`.
  - **Logic**:
    - If `fee` has a valid amount (not empty, '0', or 'PENDING'), reject as "Already Registered".
    - If `fee` is empty/PENDING, allow payment.
  - Upsert user with `fee` = 'PENDING' and `payment_channel` = '..._pending'.
- **Payment Status Sync**:
  - Update `check_payment_status`:
    - When WeChat returns `SUCCESS`:
      - **Update `fee`** to the actual configured payment amount (e.g., "0.01").
      - Update `payment_channel` to the final channel (removing '_pending').
      - Update `out_trade_no` logic to ensure we find the correct user.

### 2. Frontend Optimization (`templates/ticket.html`)
- **Dynamic Form Fields**:
  - Replace hardcoded inputs with a loop that renders fields based on `config.field_config` (managed in Admin).
  - Respect "Show", "Required", and "Label" settings.
- **QR Code Modal**:
  - Add explicit text: "请使用微信扫一扫完成支付" (Please use WeChat Scan to pay).
  - Optimize layout.
- **Payment Logic**:
  - Handle "Already Registered" error.
  - Redirect to Success Page on completion.

### 3. New Success Page
- Create `templates/success.html` to show a "Registration Successful" message.
- Add route in `main.py`.

### 4. Admin Interface
- Verify `templates/admin.html` field configuration works with the new `ticket.html` dynamic rendering.

This ensures that successful payments overwrite the 'PENDING' status with the actual amount paid.