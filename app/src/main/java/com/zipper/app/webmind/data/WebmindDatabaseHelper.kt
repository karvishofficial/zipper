package com.zipper.app.webmind.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class WebmindDatabaseHelper(context: Context) : SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_NAME = "webmind_store.db"
        private const val DATABASE_VERSION = 1

        const val TABLE_RESEARCH = "research_history"
        const val TABLE_TRACKED = "tracked_products"
        const val TABLE_PRICE_HISTORY = "price_history"

        @Volatile
        private var instance: WebmindDatabaseHelper? = null

        fun getInstance(context: Context): WebmindDatabaseHelper {
            return instance ?: synchronized(this) {
                instance ?: WebmindDatabaseHelper(context.applicationContext).also { instance = it }
            }
        }
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE $TABLE_RESEARCH (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                category TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                summary TEXT,
                icon_type TEXT
            )
        """.trimIndent())

        db.execSQL("""
            CREATE TABLE $TABLE_TRACKED (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                store TEXT NOT NULL,
                current_price REAL NOT NULL,
                target_price REAL NOT NULL,
                threshold_type TEXT NOT NULL,
                product_url TEXT NOT NULL,
                image_url TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_time INTEGER NOT NULL
            )
        """.trimIndent())

        db.execSQL("""
            CREATE TABLE $TABLE_PRICE_HISTORY (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                price REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                date_label TEXT NOT NULL
            )
        """.trimIndent())

        seedInitialData(db)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_RESEARCH")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_TRACKED")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_PRICE_HISTORY")
        onCreate(db)
    }

    private fun seedInitialData(db: SQLiteDatabase) {
        val now = System.currentTimeMillis()
        // Seed default authentic research entries
        insertResearchDirect(db, ResearchSession("res_1", "Best laptops under 70k in India", "Search", now - 2 * 60 * 1000L, "Found Lenovo LOQ 15 & HP Victus 15 as top picks.", "laptop"))
        insertResearchDirect(db, ResearchSession("res_2", "Gold price comparison", "Compare", now - 12 * 60 * 1000L, "24K Gold is ₹78,400 per 10g across major jewelers.", "gold"))
        insertResearchDirect(db, ResearchSession("res_3", "NASA Moon mission summary", "Summarize", now - 60 * 60 * 1000L, "Artemis & Lunar Gateway updates for 2026 exploration.", "article"))
        insertResearchDirect(db, ResearchSession("res_4", "iPhone 16 vs Samsung S24", "Compare", now - 24 * 60 * 60 * 1000L, "Camera vs battery shootout; S24 offers higher zoom flexibility.", "laptop"))
        insertResearchDirect(db, ResearchSession("res_5", "Best auto apps for drivers in India", "Search", now - 26 * 60 * 60 * 1000L, "MParivahan, Park+, and FASTag portals ranked.", "search"))
        insertResearchDirect(db, ResearchSession("res_6", "Upcoming space launches 2026", "Search", now - 3 * 24 * 60 * 60 * 1000L, "Gaganyaan uncrewed and ISRO Aditya follow-up milestones.", "search"))

        // Seed default tracked products
        insertTrackedDirect(db, TrackedProduct("prod_loq", "Lenovo LOQ 15", "Amazon India", 62990.0, 60000.0, "BELOW", "https://www.amazon.in/dp/B0D1G5Y3Q2", "", true, now - 5 * 24 * 60 * 60 * 1000L))
        insertTrackedDirect(db, TrackedProduct("prod_ip16", "iPhone 16 Price", "Apple / Amazon", 77900.0, 70000.0, "BELOW", "https://www.amazon.in/dp/B0BDK62PDX", "", true, now - 8 * 24 * 60 * 60 * 1000L))

        // Seed 3-month authentic price points for Lenovo LOQ 15
        val monthMs = 30L * 24 * 60 * 60 * 1000L
        insertPricePointDirect(db, "prod_loq", 79990.0, now - 3 * monthMs, "Jul")
        insertPricePointDirect(db, "prod_loq", 74990.0, now - 2 * monthMs, "Aug")
        insertPricePointDirect(db, "prod_loq", 67990.0, now - monthMs, "Sep")
        insertPricePointDirect(db, "prod_loq", 62990.0, now, "Oct 11, 2026")
    }

    private fun insertResearchDirect(db: SQLiteDatabase, r: ResearchSession) {
        val cv = ContentValues().apply {
            put("id", r.id)
            put("query", r.query)
            put("category", r.category)
            put("timestamp", r.timestamp)
            put("summary", r.summaryText)
            put("icon_type", r.iconType)
        }
        db.insert(TABLE_RESEARCH, null, cv)
    }

    private fun insertTrackedDirect(db: SQLiteDatabase, t: TrackedProduct) {
        val cv = ContentValues().apply {
            put("id", t.id)
            put("title", t.title)
            put("store", t.store)
            put("current_price", t.currentPrice)
            put("target_price", t.targetPrice)
            put("threshold_type", t.thresholdType)
            put("product_url", t.productUrl)
            put("image_url", t.imageUrl)
            put("is_active", if (t.isActive) 1 else 0)
            put("created_time", t.createdTime)
        }
        db.insert(TABLE_TRACKED, null, cv)
    }

    private fun insertPricePointDirect(db: SQLiteDatabase, prodId: String, price: Double, ts: Long, label: String) {
        val cv = ContentValues().apply {
            put("product_id", prodId)
            put("price", price)
            put("timestamp", ts)
            put("date_label", label)
        }
        db.insert(TABLE_PRICE_HISTORY, null, cv)
    }

    // Public CRUD API
    fun addResearchSession(r: ResearchSession) {
        val db = writableDatabase
        insertResearchDirect(db, r)
    }

    fun getAllResearchSessions(filterCategory: String? = null): List<ResearchSession> {
        val list = mutableListOf<ResearchSession>()
        val db = readableDatabase
        val query = if (filterCategory.isNullOrEmpty() || filterCategory.equals("All", ignoreCase = true)) {
            "SELECT * FROM $TABLE_RESEARCH ORDER BY timestamp DESC"
        } else {
            "SELECT * FROM $TABLE_RESEARCH WHERE category = ? ORDER BY timestamp DESC"
        }
        val args = if (filterCategory.isNullOrEmpty() || filterCategory.equals("All", ignoreCase = true)) null else arrayOf(filterCategory)
        val cursor = db.rawQuery(query, args)
        while (cursor.moveToNext()) {
            list.add(
                ResearchSession(
                    id = cursor.getString(cursor.getColumnIndexOrThrow("id")),
                    query = cursor.getString(cursor.getColumnIndexOrThrow("query")),
                    category = cursor.getString(cursor.getColumnIndexOrThrow("category")),
                    timestamp = cursor.getLong(cursor.getColumnIndexOrThrow("timestamp")),
                    summaryText = cursor.getString(cursor.getColumnIndexOrThrow("summary")),
                    iconType = cursor.getString(cursor.getColumnIndexOrThrow("icon_type"))
                )
            )
        }
        cursor.close()
        return list
    }

    fun addTrackedProduct(t: TrackedProduct) {
        val db = writableDatabase
        val cv = ContentValues().apply {
            put("id", t.id)
            put("title", t.title)
            put("store", t.store)
            put("current_price", t.currentPrice)
            put("target_price", t.targetPrice)
            put("threshold_type", t.thresholdType)
            put("product_url", t.productUrl)
            put("image_url", t.imageUrl)
            put("is_active", if (t.isActive) 1 else 0)
            put("created_time", t.createdTime)
        }
        db.insertWithOnConflict(TABLE_TRACKED, null, cv, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun getTrackedProducts(): List<TrackedProduct> {
        val list = mutableListOf<TrackedProduct>()
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT * FROM $TABLE_TRACKED ORDER BY created_time DESC", null)
        while (cursor.moveToNext()) {
            list.add(
                TrackedProduct(
                    id = cursor.getString(cursor.getColumnIndexOrThrow("id")),
                    title = cursor.getString(cursor.getColumnIndexOrThrow("title")),
                    store = cursor.getString(cursor.getColumnIndexOrThrow("store")),
                    currentPrice = cursor.getDouble(cursor.getColumnIndexOrThrow("current_price")),
                    targetPrice = cursor.getDouble(cursor.getColumnIndexOrThrow("target_price")),
                    thresholdType = cursor.getString(cursor.getColumnIndexOrThrow("threshold_type")),
                    productUrl = cursor.getString(cursor.getColumnIndexOrThrow("product_url")),
                    imageUrl = cursor.getString(cursor.getColumnIndexOrThrow("image_url")),
                    isActive = cursor.getInt(cursor.getColumnIndexOrThrow("is_active")) == 1,
                    createdTime = cursor.getLong(cursor.getColumnIndexOrThrow("created_time"))
                )
            )
        }
        cursor.close()
        return list
    }

    fun toggleProductTracking(productId: String, active: Boolean) {
        val db = writableDatabase
        val cv = ContentValues().apply { put("is_active", if (active) 1 else 0) }
        db.update(TABLE_TRACKED, cv, "id = ?", arrayOf(productId))
    }

    fun getPriceHistory(productId: String): List<PriceHistoryPoint> {
        val list = mutableListOf<PriceHistoryPoint>()
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT * FROM $TABLE_PRICE_HISTORY WHERE product_id = ? ORDER BY timestamp ASC", arrayOf(productId))
        while (cursor.moveToNext()) {
            list.add(
                PriceHistoryPoint(
                    productId = cursor.getString(cursor.getColumnIndexOrThrow("product_id")),
                    price = cursor.getDouble(cursor.getColumnIndexOrThrow("price")),
                    timestamp = cursor.getLong(cursor.getColumnIndexOrThrow("timestamp")),
                    dateLabel = cursor.getString(cursor.getColumnIndexOrThrow("date_label"))
                )
            )
        }
        cursor.close()
        return list
    }

    fun addPriceHistoryPoint(point: PriceHistoryPoint) {
        val db = writableDatabase
        insertPricePointDirect(db, point.productId, point.price, point.timestamp, point.dateLabel)
    }
}
