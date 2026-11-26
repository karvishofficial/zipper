import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

val secretsProps = Properties().apply {
    val secretsFile = rootProject.file("secrets.properties")
    if (secretsFile.exists()) {
        secretsFile.inputStream().use { load(it) }
    }
}

fun secretProp(key: String, default: String = ""): String {
    val raw = secretsProps.getProperty(key) ?: System.getenv(key) ?: default
    return raw.replace("\\", "\\\\").replace("\"", "\\\"")
}

android {
    namespace = "com.zipper.app"
    compileSdk = 37

    buildFeatures {
        buildConfig = true
    }

    defaultConfig {
        applicationId = "com.zipper.app"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        buildConfigField("String", "ALERT_EMAIL", "\"${secretProp("ALERT_EMAIL")}\"")
        buildConfigField("String", "SMTP_APP_PASSWORD", "\"${secretProp("SMTP_APP_PASSWORD")}\"")
        buildConfigField("String", "FIREBASE_PROJECT_ID", "\"${secretProp("FIREBASE_PROJECT_ID")}\"")
        buildConfigField("String", "FIREBASE_API_KEY", "\"${secretProp("FIREBASE_API_KEY")}\"")
        buildConfigField("String", "PRIMARY_GROQ_API_KEY", "\"${secretProp("PRIMARY_GROQ_API_KEY")}\"")
        buildConfigField("String", "SECONDARY_GROQ_API_KEY", "\"${secretProp("SECONDARY_GROQ_API_KEY")}\"")
        buildConfigField("String", "VOIPBIN_API_KEY", "\"${secretProp("VOIPBIN_API_KEY")}\"")
        buildConfigField("long", "TENCENT_SDK_APP_ID", "${secretsProps.getProperty("TENCENT_SDK_APP_ID")?.toLongOrNull() ?: 0L}L")
        buildConfigField("String", "TENCENT_SDK_SECRET_KEY", "\"${secretProp("TENCENT_SDK_SECRET_KEY")}\"")
        buildConfigField("String", "TENCENT_SECRET_ID", "\"${secretProp("TENCENT_SECRET_ID")}\"")
        buildConfigField("String", "TENCENT_SECRET_KEY", "\"${secretProp("TENCENT_SECRET_KEY")}\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

tasks.register("verifyMobileModelPackaging") {
    doLast {
        val model = project.file("src/main/assets/models/karvish_v62.onnx")
        val hardLimitBytes = 60_000_000L
        if (model.exists() && model.length() > hardLimitBytes) {
            error(
                "Mobile model packaging gate failed: ${model.name} is " +
                    "${model.length()} bytes; limit is $hardLimitBytes bytes."
            )
        }
    }
}

tasks.matching { it.name == "preReleaseBuild" }.configureEach {
    dependsOn("verifyMobileModelPackaging")
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
    implementation("com.google.android.gms:play-services-mlkit-text-recognition:19.0.0")
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    testImplementation(libs.junit)
    testImplementation("org.json:json:20231013")
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}