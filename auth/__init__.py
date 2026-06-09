"""
auth — self-contained passwordless email-OTP authentication module.

Public surface (the only things the rest of the app should import):
    from auth import auth_router, get_current_user, authenticate_ws
"""
from auth.router import auth_router
from auth.security import authenticate_ws, get_current_user

__all__ = ["auth_router", "get_current_user", "authenticate_ws"]
