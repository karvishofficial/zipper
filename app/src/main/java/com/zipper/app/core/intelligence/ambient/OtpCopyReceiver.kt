package com.zipper.app.core.intelligence.ambient

import android.content.BroadcastReceiver
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.widget.Toast

class OtpCopyReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent?) {
        val code = intent?.getStringExtra("otp_code")?.trim().orEmpty()
        if (code.isNotBlank()) {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            val clip = ClipData.newPlainText("Copied Code", code)
            clipboard?.setPrimaryClip(clip)
            Toast.makeText(context, "📋 Copied '$code' to clipboard!", Toast.LENGTH_SHORT).show()
        }
    }
}
