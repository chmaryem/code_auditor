# -- Sandbox Runner -- Auto-generated --
import sys
import os
sys.path.insert(0, r"C:\Users\Asus\Desktop\code_auditor")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(r"C:\Users\Asus\Desktop\code_auditor", ".env"))
except ImportError:
    pass

import json
import re
from pathlib import Path
from datetime import datetime

from services.code_mode_client import github, rag, kg, cache, resolver

try:
    code = '''package tn.esprit.sampleprojet;
    import java.util.Date;
    import java.util.List;
    import java.util.ArrayList;
    import jakarta.annotation.Nonnull;


    public class User {
        int id;
        String username;
        private String passwordHash;
        String email;
        private String role;
        private Date createdAt;
        private Date lastLogin;
        private boolean isActive;
        private List<String> permissions;
        private List<String> nonpermissions;

        public User(int id, String username, String passwordHash, String email,
                    String role, Date createdAt, Date lastLogin, boolean isActive) {
            this.id = id;
            this.username = username;
            this.passwordHash = passwordHash; // Assign to passwordHash
            this.email = email;
            this.role = role;
            this.createdAt = (createdAt != null) ? new Date(createdAt.getTime()) : null; // Defensive copy
            this.lastLogin = (lastLogin != null) ? new Date(lastLogin.getTime()) : null; // Defensive copy
            this.isActive = isActive; // Initialize isActive field
            this.permissions = new ArrayList<>(); // Initialize permissions list to prevent NullPointerException
            this.nonpermissions = new ArrayList<>(); // Initialize nonpermissions
        }

        public User() {
            this.permissions = new ArrayList<>();
            this.nonpermissions = new ArrayList<>();
        }

        // Getters for all private fields to maintain encapsulation
        public int getId() {
            return id;
        }

        public String getUsername() {
            return username;
        }

        public String getPasswordHash() {
            return passwordHash;
        }

        public String getEmail() {
            return email;
        }

        public String getRole() {
            return role;
        }

        public Date getCreatedAt() {
            return (createdAt != null) ? new Date(createdAt.getTime()) : null;
        }

        public Date getLastLogin() {
            return (lastLogin != null) ? new Date(lastLogin.getTime()) : null;
        }

        public boolean isActive() {
            return isActive;
        }

        public List<String> getPermissions() {
            return new ArrayList<>(permissions); // Return a defensive copy
        }

        public List<String> getNonpermissions() {
            return new ArrayList<>(nonpermissions); // Return a defensive copy
        }

        // Setter for permissions (if needed, otherwise permissions should be managed internally or via constructor)
        public void setPermissions(List<String> permissions) {
            this.permissions = (permissions != null) ? new ArrayList<>(permissions) : new ArrayList<>();
        }


        public void setNonpermissions(List<String> nonpermissions) {
            this.nonpermissions = (nonpermissions != null) ? new ArrayList<>(nonpermissions) : new ArrayList<>();
        }

        public boolean hasRole(String targetRole) {
            return this.role != null && this.role.equalsIgnoreCase(targetRole);
        }

        // 2. Permission Validation
        public boolean hasPermission(@Nonnull String permission) {
            // `permissions` list is now'''
    braces = code.count('{') == code.count('}')
    no_markers = '<<<<<<' not in code
    print('SANDBOX_OK' if braces and no_markers else 'SANDBOX_FAIL')

except Exception as _sandbox_err:
    import traceback
    print(f"SANDBOX_ERROR: {_sandbox_err}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
finally:
    try:
        github.disconnect()
    except Exception:
        pass
