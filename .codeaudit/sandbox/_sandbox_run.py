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

    import tn.esprit.sampleprojet.User;

    import javax.sql.DataSource;
    import java.sql.*;
    import java.sql.Connection;
    import java.sql.PreparedStatement;
    import java.sql.ResultSet;
    import java.sql.SQLException;
    import java.sql.Statement;
    import java.util.ArrayList;
    import java.util.List;
    import java.util.Optional;

    public class UserRepository {

        private DataSource dataSource;

        public UserRepository(DataSource dataSource) {
            this.dataSource = dataSource;
        }
        private String hashPassword(String plainPassword) {
            return "hashed_" + plainPassword; // Example placeholder
        }

        public User findById(int id) throws SQLException {
            String sql = "SELECT id, username, email FROM users WHERE id = ?";
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement pstmt = conn.prepareStatement(sql)) {
                pstmt.setInt(1, id);
                try (ResultSet rs = pstmt.executeQuery()) {
                    if (rs.next()) {
                        User user = new User();
                        user.id = rs.getInt("id");
                        user.username = rs.getString("username");
                        user.email = rs.getString("email");
                        return user;
                    }
                }
            }
            return null;
        }

    public List<User> findAll() throws SQLException {
            List<User> users = new ArrayList<>();
            String sql = "SELECT id, username, email FROM users";
            try (Connection conn = dataSource.getConnection();
                 Statement stmt = conn.createStatement();
                 ResultSet rs = stmt.executeQuery(sql)) {

                while (rs.next()) {
                    User user = new User();
                    user.id = rs.getInt("id");
                    user.username = rs.getString("username");
                    user.email = rs.getString("email");
                    users.add(user);
                }
            }

            return users;
        }
        public void save(User user) throws SQLException {
            String sql = "INSERT INTO users (username, email, password) VALUES (?, ?, ?)";
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement pstmt = conn.prepareStatement(sql)) {

                pstmt.setString(1, user.username);
                pstmt.setString(2, user.email);
                pstmt.setString(3, hashPassword(user.getPasswordHash())); // Hash password before saving

                pstmt.executeUpdate();
            }
        }



        public int countUsers() throws SQLException {
            // Changement de la requête de COUNT(*) à COUNT(1)
            String sql = "SELECT COUNT(1) AS total_count FROM users";
            try (Connection conn = dataSource.getConnection();
                 Statement stmt = conn.createStatement();
                 ResultSet rs = stmt.executeQuery(sql)) {

                if (rs.next()) {
                    return rs.getInt("total");
                }
            }
            return 0;
        }

    public void batchInsert(List<User> users) throws SQLExce'''
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
