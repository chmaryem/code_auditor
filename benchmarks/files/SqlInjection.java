package tn.esprit.benchmarks;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;

public class SqlInjection {
    public void getUser(Connection conn, String userId) throws Exception {
        Statement stmt = conn.createStatement();
        String query = "SELECT * FROM users WHERE id = '" + userId + "'";
        ResultSet rs = stmt.executeQuery(query);
        while (rs.next()) {
            System.out.println("User: " + rs.getString("username"));
        }
    }
}
