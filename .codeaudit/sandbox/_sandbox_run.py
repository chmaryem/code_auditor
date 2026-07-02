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

    import org.springframework.beans.factory.annotation.Autowired;
    import org.springframework.stereotype.Service;
    import java.sql.*;
    import java.sql.Connection;
    import java.sql.PreparedStatement;
    import java.sql.ResultSet;
    import java.sql.SQLException;
    import java.sql.Timestamp; // Added for handling Date/Timestamp conversion
    import java.util.*;
    import java.util.ArrayList;
    import java.util.Date;
    import java.util.Date; // Explicitly imported for Date objects
    import java.util.List;
    import javax.sql.DataSource;
    import tn.esprit.sampleprojet.User;

    @Service
    public class UserService {

    private final DataSource dataSource;

    @Autowired
    public UserService(DataSource dataSource) {
        this.dataSource = dataSource;
    }
        public User findByUsername(String username) throws SQLException {
    // All fields required for a complete User object (as per User constructor) should be retrieved.
    String query = "SELECT id, username, password_hash, email, role, created_at, last_login, is_active FROM users WHERE username = ?";
    // Parameterized query to prevent SQL injection
    PreparedStatement statement = connection.prepareStatement(query);
    statement.setString(1, username);
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement stmt = conn.prepareStatement(query)) {
                try (ResultSet rs = stmt.executeQuery()) {
                    if (rs.next()) {
    int id = rs.getInt("id");
    String retrievedUsername = rs.getString("username");
    String passwordHash = rs.getString("password_hash");
    String email = rs.getString("email");
    String role = rs.getString("role");
    Timestamp createdAtTimestamp = rs.getTimestamp("created_at");
    Timestamp lastLoginTimestamp = rs.getTimestamp("last_login");
    boolean isActive = rs.getBoolean("is_active");

    Date createdAt = (createdAtTimestamp != null) ? new Date(createdAtTimestamp.getTime()) : null;
    Date lastLogin = (lastLoginTimestamp != null) ? new Date(lastLoginTimestamp.getTime()) : null;

    return mapUser(new User(id, retrievedUsername, passwordHash, email, role, createdAt, lastLogin, isActive));
                    }
                }
            }
            return null;
        }

        public boolean authenticate(String username, String password) throws SQLException {
    ```
            String query = "SELECT password_hash FROM users WHERE username = ?";
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement stmt = conn.prepareStatement(query)) {
                stmt.setString(1, username);
                try (ResultSet rs = stmt.executeQuery()) {
                    if (rs.next()) {
    String storedPasswordHash = rs.getString("password_hash");
    return hashPassword(password).equals(storedPasswordHash);
                    }
                }
            }
            return false;
        }

        public User createUser(String username, String email, String password, String role) throws SQLException {
            String hashedPassword = hashPassword(password);
            String insertQuery = "INSERT INTO users (username, emai'''
    # Quick syntax check
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
