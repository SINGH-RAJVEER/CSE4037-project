import { Database } from "bun:sqlite";
import { drizzle } from "drizzle-orm/bun-sqlite";
import { migrate } from "drizzle-orm/bun-sqlite/migrator";
import * as schema from "./schema";

const dbPath = process.env.DATABASE_URL?.replace(/^file:/, "") ?? (process.env.NODE_ENV === "production" ? "/tmp/chess.db" : "chess.db");
const database = new Database(dbPath);
const db = drizzle(database, { schema });

try {
  migrate(db, { migrationsFolder: "drizzle" });
  console.log("Database migrated successfully");
} catch (error) {
  console.error("Database migration failed:", error);
}

export { db, schema };
