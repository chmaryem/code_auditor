package tn.esprit.benchmarks;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

public class ResourceLeak {
    public void processData() throws Exception {
        Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/db", "user", "pass");
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery("SELECT * FROM data");
        
        // ResultSet and Statement are never closed.
        while (rs.next()) {
            System.out.println(rs.getString(1));
        }
    }
}
