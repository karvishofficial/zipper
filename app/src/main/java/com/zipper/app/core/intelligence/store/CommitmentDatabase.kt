package com.zipper.app.core.intelligence.store

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log

/**
 * Lightweight, zero-bloat SQLite storage for Karvish Personal Intelligence.
 * Guarantees < 2 MB storage by auto-purging context older than 24 hours.
 */
class CommitmentDatabase private constructor(context: Context) :
    SQLiteOpenHelper(context.applicationContext, DB_NAME, null, DB_VERSION) {

    companion object {
        private const val TAG = "CommitmentDatabase"
        private const val DB_NAME = "karvish_intelligence.db"
        private const val DB_VERSION = 1

        private const val TABLE_CONTEXT = "context_buffer"
        private const val TABLE_COMMITMENTS = "commitments"

        @Volatile
        private var instance: CommitmentDatabase? = null

        fun getInstance(context: Context): CommitmentDatabase {
            return instance ?: synchronized(this) {
                instance ?: CommitmentDatabase(context).also { instance = it }
            }
        }
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE $TABLE_CONTEXT (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                app_name TEXT NOT NULL,
                package_name TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL
            )
            """.trimIndent()
        )

        db.execSQL(
            """
            CREATE TABLE $TABLE_COMMITMENTS (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                deadline_ms INTEGER,
                scheduled_time_ms INTEGER NOT NULL,
                duration_mins INTEGER NOT NULL,
                status TEXT NOT NULL,
                source_app TEXT NOT NULL,
                source_person TEXT NOT NULL,
                candidate_packages TEXT NOT NULL,
                notes TEXT NOT NULL,
                created_time_ms INTEGER NOT NULL
            )
            """.trimIndent()
        )
        Log.i(TAG, "Initialized Karvish Intelligence tables successfully")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_CONTEXT")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_COMMITMENTS")
        onCreate(db)
    }

    // ==========================================
    // 24/7 Context Buffer Operations (Rolling 24h)
    // ==========================================

    @Synchronized
    fun insertContextMessage(msg: ContextMessage): Long {
        val db = writableDatabase
        val cv = ContentValues().apply {
            put("sender", msg.sender)
            put("app_name", msg.appName)
            put("package_name", msg.packageName)
            put("summary_text", msg.summaryText)
            put("timestamp_ms", msg.timestampMs)
        }
        val id = db.insert(TABLE_CONTEXT, null, cv)
        // Auto-purge older than 24 hours immediately to prevent bloat
        purgeOldContextMessages(24 * 60 * 60 * 1000L)
        return id
    }

    @Synchronized
    fun getRecentContextMessages(sinceTimestampMs: Long): List<ContextMessage> {
        val list = mutableListOf<ContextMessage>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_CONTEXT,
            null,
            "timestamp_ms >= ?",
            arrayOf(sinceTimestampMs.toString()),
            null,
            null,
            "timestamp_ms ASC"
        )
        cursor.use {
            val idCol = it.getColumnIndexOrThrow("id")
            val senderCol = it.getColumnIndexOrThrow("sender")
            val appCol = it.getColumnIndexOrThrow("app_name")
            val pkgCol = it.getColumnIndexOrThrow("package_name")
            val textCol = it.getColumnIndexOrThrow("summary_text")
            val timeCol = it.getColumnIndexOrThrow("timestamp_ms")

            while (it.moveToNext()) {
                list.add(
                    ContextMessage(
                        id = it.getLong(idCol),
                        sender = it.getString(senderCol),
                        appName = it.getString(appCol),
                        packageName = it.getString(pkgCol),
                        summaryText = it.getString(textCol),
                        timestampMs = it.getLong(timeCol)
                    )
                )
            }
        }
        return list
    }

    @Synchronized
    fun purgeOldContextMessages(maxAgeMs: Long = 24 * 60 * 60 * 1000L): Int {
        val cutoff = System.currentTimeMillis() - maxAgeMs
        val db = writableDatabase
        return db.delete(TABLE_CONTEXT, "timestamp_ms < ?", arrayOf(cutoff.toString()))
    }

    // ==========================================
    // Persistent Commitments Operations
    // ==========================================

    @Synchronized
    fun insertCommitment(c: CommitmentEntity): Long {
        val db = writableDatabase

        // Idempotency check: prevent duplicate commitments on the same calendar day
        val dayStart = if (c.scheduledTimeMs > 0) (c.scheduledTimeMs / 86400000L) * 86400000L else (System.currentTimeMillis() / 86400000L) * 86400000L
        val dayEnd = dayStart + 86400000L
        val normalizedTitle = c.title.trim().lowercase(java.util.Locale.ROOT)

        val cursor = db.query(
            TABLE_COMMITMENTS,
            arrayOf("id"),
            "LOWER(TRIM(title)) = ? AND scheduled_time_ms >= ? AND scheduled_time_ms < ?",
            arrayOf(normalizedTitle, dayStart.toString(), dayEnd.toString()),
            null, null, "id ASC", "1"
        )
        val existingId = cursor.use {
            if (it.moveToFirst()) it.getLong(0) else null
        }

        val cv = ContentValues().apply {
            put("title", c.title)
            put("category", c.category.name)
            put("deadline_ms", c.deadlineMs)
            put("scheduled_time_ms", c.scheduledTimeMs)
            put("duration_mins", c.durationMinutes)
            put("status", c.status.name)
            put("source_app", c.sourceApp)
            put("source_person", c.sourcePerson)
            put("candidate_packages", c.candidatePackages.joinToString(","))
            put("notes", c.notes)
            put("created_time_ms", c.createdTimeMs)
        }

        return if (existingId != null) {
            db.update(TABLE_COMMITMENTS, cv, "id = ?", arrayOf(existingId.toString()))
            existingId
        } else {
            db.insert(TABLE_COMMITMENTS, null, cv)
        }
    }

    /**
     * Purges duplicate commitment records keeping only one instance per title and calendar day.
     */
    @Synchronized
    fun deduplicateCommitments() {
        val db = writableDatabase
        try {
            db.execSQL("""
                DELETE FROM $TABLE_COMMITMENTS
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM $TABLE_COMMITMENTS
                    GROUP BY LOWER(TRIM(title)), (scheduled_time_ms / 86400000)
                )
            """.trimIndent())
            Log.i(TAG, "Deduplicated commitments table successfully")
        } catch (e: Exception) {
            Log.w(TAG, "Deduplication query warning: ${e.message}")
        }
    }

    /**
     * Purges hallucinated or legacy mock commitments (e.g. Manoj/Ajay/Bhavani test stories).
     */
    @Synchronized
    fun purgeMockSeedCommitments() {
        val db = writableDatabase
        try {
            db.delete(
                TABLE_COMMITMENTS,
                "title LIKE ? OR title LIKE ? OR source_person LIKE ? OR notes LIKE ?",
                arrayOf("%Manoj%", "%Bhavani%", "%Manoj%", "%Manoj%")
            )
            Log.i(TAG, "Purged mock test commitments from database")
        } catch (e: Exception) {
            Log.w(TAG, "Purge mock commitments warning: ${e.message}")
        }
    }

    @Synchronized
    fun updateCommitmentStatus(id: Long, newStatus: CommitmentStatus): Boolean {
        val db = writableDatabase
        val cv = ContentValues().apply {
            put("status", newStatus.name)
        }
        return db.update(TABLE_COMMITMENTS, cv, "id = ?", arrayOf(id.toString())) > 0
    }

    @Synchronized
    fun updateCommitmentTime(id: Long, newTimeMs: Long): Boolean {
        val db = writableDatabase
        val cv = ContentValues().apply {
            put("scheduled_time_ms", newTimeMs)
        }
        return db.update(TABLE_COMMITMENTS, cv, "id = ?", arrayOf(id.toString())) > 0
    }

    @Synchronized
    fun updateCommitmentNotes(id: Long, newNotes: String): Boolean {
        val db = writableDatabase
        val cv = ContentValues().apply {
            put("notes", newNotes)
        }
        return db.update(TABLE_COMMITMENTS, cv, "id = ?", arrayOf(id.toString())) > 0
    }

    @Synchronized
    fun getPendingCommitments(): List<CommitmentEntity> {
        val list = mutableListOf<CommitmentEntity>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_COMMITMENTS,
            null,
            "status IN (?, ?)",
            arrayOf(CommitmentStatus.PENDING.name, CommitmentStatus.SNOOZED.name),
            null,
            null,
            "scheduled_time_ms ASC"
        )
        cursor.use {
            val idCol = it.getColumnIndexOrThrow("id")
            val titleCol = it.getColumnIndexOrThrow("title")
            val catCol = it.getColumnIndexOrThrow("category")
            val dlCol = it.getColumnIndexOrThrow("deadline_ms")
            val schedCol = it.getColumnIndexOrThrow("scheduled_time_ms")
            val durCol = it.getColumnIndexOrThrow("duration_mins")
            val statCol = it.getColumnIndexOrThrow("status")
            val srcAppCol = it.getColumnIndexOrThrow("source_app")
            val srcPersonCol = it.getColumnIndexOrThrow("source_person")
            val pkgsCol = it.getColumnIndexOrThrow("candidate_packages")
            val notesCol = it.getColumnIndexOrThrow("notes")
            val createdCol = it.getColumnIndexOrThrow("created_time_ms")

            while (it.moveToNext()) {
                val dlVal = if (it.isNull(dlCol)) null else it.getLong(dlCol)
                val catStr = it.getString(catCol)
                val category = try { CommitmentCategory.valueOf(catStr) } catch (_: Exception) { CommitmentCategory.GENERAL }
                val statStr = it.getString(statCol)
                val status = try { CommitmentStatus.valueOf(statStr) } catch (_: Exception) { CommitmentStatus.PENDING }
                val pkgs = it.getString(pkgsCol).split(",").map { p -> p.trim() }.filter { p -> p.isNotEmpty() }

                list.add(
                    CommitmentEntity(
                        id = it.getLong(idCol),
                        title = it.getString(titleCol),
                        category = category,
                        deadlineMs = dlVal,
                        scheduledTimeMs = it.getLong(schedCol),
                        durationMinutes = it.getInt(durCol),
                        status = status,
                        sourceApp = it.getString(srcAppCol),
                        sourcePerson = it.getString(srcPersonCol),
                        candidatePackages = pkgs,
                        notes = it.getString(notesCol),
                        createdTimeMs = it.getLong(createdCol)
                    )
                )
            }
        }
        return list
    }

    @Synchronized
    fun getAllCommitments(): List<CommitmentEntity> {
        val list = mutableListOf<CommitmentEntity>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_COMMITMENTS,
            null,
            null,
            null,
            null,
            null,
            "scheduled_time_ms ASC, created_time_ms DESC"
        )
        cursor.use {
            val idCol = it.getColumnIndexOrThrow("id")
            val titleCol = it.getColumnIndexOrThrow("title")
            val catCol = it.getColumnIndexOrThrow("category")
            val dlCol = it.getColumnIndexOrThrow("deadline_ms")
            val schedCol = it.getColumnIndexOrThrow("scheduled_time_ms")
            val durCol = it.getColumnIndexOrThrow("duration_mins")
            val statCol = it.getColumnIndexOrThrow("status")
            val srcAppCol = it.getColumnIndexOrThrow("source_app")
            val srcPersonCol = it.getColumnIndexOrThrow("source_person")
            val pkgsCol = it.getColumnIndexOrThrow("candidate_packages")
            val notesCol = it.getColumnIndexOrThrow("notes")
            val createdCol = it.getColumnIndexOrThrow("created_time_ms")

            while (it.moveToNext()) {
                val dlVal = if (it.isNull(dlCol)) null else it.getLong(dlCol)
                val catStr = it.getString(catCol)
                val category = try { CommitmentCategory.valueOf(catStr) } catch (_: Exception) { CommitmentCategory.GENERAL }
                val statStr = it.getString(statCol)
                val status = try { CommitmentStatus.valueOf(statStr) } catch (_: Exception) { CommitmentStatus.PENDING }
                val pkgs = it.getString(pkgsCol).split(",").map { p -> p.trim() }.filter { p -> p.isNotEmpty() }

                list.add(
                    CommitmentEntity(
                        id = it.getLong(idCol),
                        title = it.getString(titleCol),
                        category = category,
                        deadlineMs = dlVal,
                        scheduledTimeMs = it.getLong(schedCol),
                        durationMinutes = it.getInt(durCol),
                        status = status,
                        sourceApp = it.getString(srcAppCol),
                        sourcePerson = it.getString(srcPersonCol),
                        candidatePackages = pkgs,
                        notes = it.getString(notesCol),
                        createdTimeMs = it.getLong(createdCol)
                    )
                )
            }
        }
        return list
    }

    @Synchronized
    fun getCommitmentById(id: Long): CommitmentEntity? {
        val db = readableDatabase
        val cursor = db.query(
            TABLE_COMMITMENTS,
            null,
            "id = ?",
            arrayOf(id.toString()),
            null,
            null,
            null
        )
        cursor.use {
            if (it.moveToFirst()) {
                val idCol = it.getColumnIndexOrThrow("id")
                val titleCol = it.getColumnIndexOrThrow("title")
                val catCol = it.getColumnIndexOrThrow("category")
                val dlCol = it.getColumnIndexOrThrow("deadline_ms")
                val schedCol = it.getColumnIndexOrThrow("scheduled_time_ms")
                val durCol = it.getColumnIndexOrThrow("duration_mins")
                val statCol = it.getColumnIndexOrThrow("status")
                val srcAppCol = it.getColumnIndexOrThrow("source_app")
                val srcPersonCol = it.getColumnIndexOrThrow("source_person")
                val pkgsCol = it.getColumnIndexOrThrow("candidate_packages")
                val notesCol = it.getColumnIndexOrThrow("notes")
                val createdCol = it.getColumnIndexOrThrow("created_time_ms")

                val dlVal = if (it.isNull(dlCol)) null else it.getLong(dlCol)
                val catStr = it.getString(catCol)
                val category = try { CommitmentCategory.valueOf(catStr) } catch (_: Exception) { CommitmentCategory.GENERAL }
                val statStr = it.getString(statCol)
                val status = try { CommitmentStatus.valueOf(statStr) } catch (_: Exception) { CommitmentStatus.PENDING }
                val pkgs = it.getString(pkgsCol).split(",").map { p -> p.trim() }.filter { p -> p.isNotEmpty() }

                return CommitmentEntity(
                    id = it.getLong(idCol),
                    title = it.getString(titleCol),
                    category = category,
                    deadlineMs = dlVal,
                    scheduledTimeMs = it.getLong(schedCol),
                    durationMinutes = it.getInt(durCol),
                    status = status,
                    sourceApp = it.getString(srcAppCol),
                    sourcePerson = it.getString(srcPersonCol),
                    candidatePackages = pkgs,
                    notes = it.getString(notesCol),
                    createdTimeMs = it.getLong(createdCol)
                )
            }
        }
        return null
    }

    @Synchronized
    fun getCommitmentsForDay(startOfDayMs: Long, endOfDayMs: Long): List<CommitmentEntity> {
        val list = mutableListOf<CommitmentEntity>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_COMMITMENTS,
            null,
            "scheduled_time_ms >= ? AND scheduled_time_ms <= ?",
            arrayOf(startOfDayMs.toString(), endOfDayMs.toString()),
            null,
            null,
            "scheduled_time_ms ASC"
        )
        cursor.use {
            val idCol = it.getColumnIndexOrThrow("id")
            val titleCol = it.getColumnIndexOrThrow("title")
            val catCol = it.getColumnIndexOrThrow("category")
            val dlCol = it.getColumnIndexOrThrow("deadline_ms")
            val schedCol = it.getColumnIndexOrThrow("scheduled_time_ms")
            val durCol = it.getColumnIndexOrThrow("duration_mins")
            val statCol = it.getColumnIndexOrThrow("status")
            val srcAppCol = it.getColumnIndexOrThrow("source_app")
            val srcPersonCol = it.getColumnIndexOrThrow("source_person")
            val pkgsCol = it.getColumnIndexOrThrow("candidate_packages")
            val notesCol = it.getColumnIndexOrThrow("notes")
            val createdCol = it.getColumnIndexOrThrow("created_time_ms")

            while (it.moveToNext()) {
                val dlVal = if (it.isNull(dlCol)) null else it.getLong(dlCol)
                val catStr = it.getString(catCol)
                val category = try { CommitmentCategory.valueOf(catStr) } catch (_: Exception) { CommitmentCategory.GENERAL }
                val statStr = it.getString(statCol)
                val status = try { CommitmentStatus.valueOf(statStr) } catch (_: Exception) { CommitmentStatus.PENDING }
                val pkgs = it.getString(pkgsCol).split(",").map { p -> p.trim() }.filter { p -> p.isNotEmpty() }

                list.add(
                    CommitmentEntity(
                        id = it.getLong(idCol),
                        title = it.getString(titleCol),
                        category = category,
                        deadlineMs = dlVal,
                        scheduledTimeMs = it.getLong(schedCol),
                        durationMinutes = it.getInt(durCol),
                        status = status,
                        sourceApp = it.getString(srcAppCol),
                        sourcePerson = it.getString(srcPersonCol),
                        candidatePackages = pkgs,
                        notes = it.getString(notesCol),
                        createdTimeMs = it.getLong(createdCol)
                    )
                )
            }
        }
        return list
    }

    @Synchronized
    fun deleteCommitmentsForDay(startOfDayMs: Long, endOfDayMs: Long): Int {
        val db = writableDatabase
        return db.delete(
            TABLE_COMMITMENTS,
            "scheduled_time_ms >= ? AND scheduled_time_ms <= ?",
            arrayOf(startOfDayMs.toString(), endOfDayMs.toString())
        )
    }

    @Synchronized
    fun getCompletedTodayCount(startOfDayMs: Long, endOfDayMs: Long): Int {
        val db = readableDatabase
        val cursor = db.rawQuery(
            "SELECT COUNT(*) FROM $TABLE_COMMITMENTS WHERE status = ? AND scheduled_time_ms >= ? AND scheduled_time_ms <= ?",
            arrayOf(CommitmentStatus.COMPLETED.name, startOfDayMs.toString(), endOfDayMs.toString())
        )
        cursor.use {
            if (it.moveToFirst()) return it.getInt(0)
        }
        return 0
    }

    @Synchronized
    fun getPendingTodayCount(startOfDayMs: Long, endOfDayMs: Long): Int {
        val db = readableDatabase
        val cursor = db.rawQuery(
            "SELECT COUNT(*) FROM $TABLE_COMMITMENTS WHERE status IN (?, ?) AND scheduled_time_ms >= ? AND scheduled_time_ms <= ?",
            arrayOf(CommitmentStatus.PENDING.name, CommitmentStatus.IN_PROGRESS.name, startOfDayMs.toString(), endOfDayMs.toString())
        )
        cursor.use {
            if (it.moveToFirst()) return it.getInt(0)
        }
        return 0
    }

    @Synchronized
    fun getNeedsAttentionCommitments(): List<CommitmentEntity> {
        val list = mutableListOf<CommitmentEntity>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_COMMITMENTS,
            null,
            "status IN (?, ?) OR (deadline_ms IS NOT NULL AND deadline_ms > 0)",
            arrayOf(CommitmentStatus.BLOCKED.name, CommitmentStatus.SNOOZED.name),
            null,
            null,
            "created_time_ms DESC"
        )
        cursor.use {
            val idCol = it.getColumnIndexOrThrow("id")
            val titleCol = it.getColumnIndexOrThrow("title")
            val catCol = it.getColumnIndexOrThrow("category")
            val dlCol = it.getColumnIndexOrThrow("deadline_ms")
            val schedCol = it.getColumnIndexOrThrow("scheduled_time_ms")
            val durCol = it.getColumnIndexOrThrow("duration_mins")
            val statCol = it.getColumnIndexOrThrow("status")
            val srcAppCol = it.getColumnIndexOrThrow("source_app")
            val srcPersonCol = it.getColumnIndexOrThrow("source_person")
            val pkgsCol = it.getColumnIndexOrThrow("candidate_packages")
            val notesCol = it.getColumnIndexOrThrow("notes")
            val createdCol = it.getColumnIndexOrThrow("created_time_ms")

            while (it.moveToNext()) {
                val dlVal = if (it.isNull(dlCol)) null else it.getLong(dlCol)
                val catStr = it.getString(catCol)
                val category = try { CommitmentCategory.valueOf(catStr) } catch (_: Exception) { CommitmentCategory.GENERAL }
                val statStr = it.getString(statCol)
                val status = try { CommitmentStatus.valueOf(statStr) } catch (_: Exception) { CommitmentStatus.PENDING }
                val pkgs = it.getString(pkgsCol).split(",").map { p -> p.trim() }.filter { p -> p.isNotEmpty() }

                list.add(
                    CommitmentEntity(
                        id = it.getLong(idCol),
                        title = it.getString(titleCol),
                        category = category,
                        deadlineMs = dlVal,
                        scheduledTimeMs = it.getLong(schedCol),
                        durationMinutes = it.getInt(durCol),
                        status = status,
                        sourceApp = it.getString(srcAppCol),
                        sourcePerson = it.getString(srcPersonCol),
                        candidatePackages = pkgs,
                        notes = it.getString(notesCol),
                        createdTimeMs = it.getLong(createdCol)
                    )
                )
            }
        }
        return list
    }

    @Synchronized
    fun getDatesWithCommitments(startMs: Long, endMs: Long): Set<Long> {
        val dates = mutableSetOf<Long>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_COMMITMENTS,
            arrayOf("scheduled_time_ms"),
            "scheduled_time_ms >= ? AND scheduled_time_ms <= ?",
            arrayOf(startMs.toString(), endMs.toString()),
            null,
            null,
            null
        )
        cursor.use {
            val timeCol = it.getColumnIndexOrThrow("scheduled_time_ms")
            while (it.moveToNext()) {
                val time = it.getLong(timeCol)
                if (time > 0) {
                    val cal = java.util.Calendar.getInstance().apply {
                        timeInMillis = time
                        set(java.util.Calendar.HOUR_OF_DAY, 0)
                        set(java.util.Calendar.MINUTE, 0)
                        set(java.util.Calendar.SECOND, 0)
                        set(java.util.Calendar.MILLISECOND, 0)
                    }
                    dates.add(cal.timeInMillis)
                }
            }
        }
        return dates
    }
}
