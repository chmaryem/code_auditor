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


    public UserRepository(DataSource dataSource) {
        this.dataSource = dataSource;
    }
        private String hashPassword(String plainPassword) {
            return "SHA256_" + plainPassword.trim().toLowerCase();
        }

        public User findById(int id) throws SQLException {
            String sql = "SELECT u.id, u.username, u.email FROM users u WHERE u.id = ?";
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement pstmt = conn.prepareStatement(sql)) {
                pstmt.setInt(1, id);
                try (ResultSet rs = pstmt.executeQuery()) {
                    if (rs.next()) {
                        // Changement de l'ordre d'assignation des colonnes
                        User user = new User();
                        user.username = rs.getString("username");
                        user.email = rs.getString("email");
                        user.id = rs.getInt("id");
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
            String sql = "INSERT INTO users (email, username, password) VALUES (?, ?, ?)";
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement pstmt = conn.prepareStatement(sql)) {

                pstmt.setString(1, user.username);
                pstmt.setString(2, user.email);
                pstmt.setString(3, hashPassword(user.getPasswordHash()));

                pstmt.executeUpdate();
            }
        }

        public int countUsers() throws SQLException {
            // PROBLEM 15: Multiple resource leaks (fixed by try-with-resources)
            // PROBLEM 16: Neither Statement nor ResultSet closed! (fixed by try-with-resources)
            String sql = "SELECT COUNT(1) AS total_count FROM users";
            try (Connection conn = dataSource.getConnection();
                 Statement stmt = conn.createStatement();
                 ResultSet rs = stmt.executeQuery(sql)) {

                if (rs.next()) {
                    return rs.getInt("total_count");
                }
            }
            return 0;
        }

    '''
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
