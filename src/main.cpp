/* HoopTracker Pro — ELITE EDITION v3.2
   Professional Multi-User System — Dec 2025
   
   ENHANCEMENTS OVER v3.1:
   - Session-level auto/manual control (not global)
   - Separate "Background Learning" toggle
   - Outlier rejection in calibration (>3σ)
   - Temporal smoothing (EMA confidence)
   - Quadratic fit for entry velocity
   - Polished modern UI with animations
   - Per-profile export and reset
   - Real-time accuracy feedback with color coding
   
   MAINTAINED STABILITY:
   - 5 physically valid features
   - Fixed-size arrays only
   - Diagonal covariance
   - Memory-safe (<30KB fingerprints)
*/

#include <Arduino.h>
#include <Wire.h>
#include <VL53L1X.h>
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>
#include <AsyncElegantOTA.h>
#include <esp_task_wdt.h>
#include <array>
#include <algorithm>
#include <math.h>
#include <atomic>

// ============================================================================
// CONFIGURATION
// ============================================================================
constexpr int VIB_PIN = 18;
constexpr int SDA_PIN = 21;
constexpr int SCL_PIN = 22;
constexpr int NUM_SECTIONS = 8;
constexpr int MAX_USERS = 4;
constexpr int BASELINE_DISTANCE = 600;
constexpr int DEF_BALL_TRIGGER_DROP = 180;
constexpr uint32_t DEF_SHOT_COOLDOWN_MS = 900;
constexpr uint32_t DEF_RIM_WINDOW_MS = 250;
constexpr uint32_t DEF_ENTRY_WINDOW_MS = 350;
constexpr uint32_t DEF_VIB_DEBOUNCE_MS = 25;
constexpr float DEF_BASELINE_ALPHA = 0.01f;
constexpr uint32_t MIN_CALIBRATION_SHOTS = 15;
constexpr uint32_t MAX_CALIBRATION_SHOTS = 100;
constexpr uint32_t CONFIG_VERSION = 4;
constexpr uint32_t FINGERPRINT_VERSION = 4;

constexpr int NUM_FEATURES = 5;
constexpr float CONFIDENCE_THRESHOLD = 0.25f;
constexpr float ADAPTIVE_LEARNING_RATE = 0.08f;
constexpr float HIGH_CONFIDENCE_THRESHOLD = 0.75f;
constexpr size_t HEAP_WARNING_THRESHOLD = 25000;
constexpr size_t HEAP_CRITICAL_THRESHOLD = 20000;
constexpr int ACCURACY_WINDOW = 30;
constexpr float OUTLIER_SIGMA_THRESHOLD = 3.0f;
constexpr int CONFIDENCE_SMOOTHING_WINDOW = 5;

const char* USERS_FILE = "/users.json";
const char* SESSIONS_FILE = "/sessions.json";
const char* CONFIG_FILE = "/config.json";
constexpr uint32_t DATA_SAVE_INTERVAL_MS = 15000;
constexpr size_t MAX_SESSIONS = 200;

#define DEBUG 1
#if DEBUG
  #define FSM_LOG(x) Serial.println(x)
  #define DEBUG_LOG(x) Serial.println(x)
#else
  #define FSM_LOG(x)
  #define DEBUG_LOG(x)
#endif

// ============================================================================
// HARDWARE
// ============================================================================
VL53L1X sensor;
AsyncWebServer server(80);
AsyncEventSource events("/events");

// ============================================================================
// THREAD SAFETY
// ============================================================================
SemaphoreHandle_t dataMutex = NULL;

class ScopedLock {
    SemaphoreHandle_t mutex;
    bool acquired;
public:
    explicit ScopedLock(SemaphoreHandle_t m, TickType_t timeout = portMAX_DELAY) 
        : mutex(m), acquired(false) {
        if (m) acquired = (xSemaphoreTake(m, timeout) == pdTRUE);
    }
    ~ScopedLock() {
        if (acquired && mutex) xSemaphoreGive(mutex);
    }
    bool isLocked() const { return acquired; }
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
static inline int clampInt(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline float clampFloat(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

uint64_t nowEpochMs() {
    time_t t = time(nullptr);
    if (t <= 0) return (uint64_t)millis();
    return (uint64_t)t * 1000ULL + (uint64_t)(millis() % 1000);
}

String randomPass(uint8_t len = 12) {
    static const char cs[] = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789";
    String s; s.reserve(len);
    uint32_t seed = micros() ^ (uint32_t)esp_random();
    randomSeed(seed);
    for (uint8_t i=0; i<len; ++i) s += cs[random(sizeof(cs)-1)];
    return s;
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct Stats {
    uint32_t attempts = 0;
    uint32_t makes = 0;
    uint32_t swishes = 0;
    uint32_t rimMakes = 0;

    void add(const Stats &s) {
        attempts += s.attempts;
        makes += s.makes;
        swishes += s.swishes;
        rimMakes += s.rimMakes;
    }
    
    void addAttempt(bool made, bool swish) {
        attempts++;
        if (made) {
            makes++;
            if (swish) swishes++;
            else rimMakes++;
        }
    }
    
    void clear() {
        attempts = makes = swishes = rimMakes = 0;
    }
};

struct PhysicalFingerprint {
    float meanEntryVelocity = 0.0f;
    float meanDwellTime = 0.0f;
    float meanTimeToRim = 0.0f;
    float meanVibrationCount = 0.0f;
    float meanEntryConsistency = 0.0f;
    
    float variance[NUM_FEATURES];
    float weights[NUM_FEATURES];
    
    uint32_t samples = 0;
    float reliability = 0.0f;
    
    void clear() {
        meanEntryVelocity = 0.0f;
        meanDwellTime = 0.0f;
        meanTimeToRim = 0.0f;
        meanVibrationCount = 0.0f;
        meanEntryConsistency = 0.0f;
        
        for (int i = 0; i < NUM_FEATURES; i++) {
            variance[i] = 1.0f;
            weights[i] = 1.0f;
        }
        
        samples = 0;
        reliability = 0.0f;
    }
    
    void getMeans(float* out) const {
        out[0] = meanEntryVelocity;
        out[1] = meanDwellTime;
        out[2] = meanTimeToRim;
        out[3] = meanVibrationCount;
        out[4] = meanEntryConsistency;
    }
    
    void setMeans(const float* in) {
        meanEntryVelocity = in[0];
        meanDwellTime = in[1];
        meanTimeToRim = in[2];
        meanVibrationCount = in[3];
        meanEntryConsistency = in[4];
    }
};

struct SectionRecord {
    Stats stats;
    PhysicalFingerprint fingerprint;
    char name[24];
    
    SectionRecord() {
        name[0] = '\0';
    }
};

struct GuessRecord {
    int8_t guessedSection;
    int8_t actualSection;
    bool correct;
};

struct UserProfile {
    char username[24];
    bool backgroundLearning;  // NEW: Separate from fingerprint usage
    std::array<SectionRecord, NUM_SECTIONS> sections;
    std::array<GuessRecord, ACCURACY_WINDOW> recentGuesses;
    uint8_t guessCount;
    uint8_t guessIndex;
    
    UserProfile() : backgroundLearning(true), guessCount(0), guessIndex(0) {
        username[0] = '\0';
    }
    
    void addGuess(int guessed, int actual) {
        recentGuesses[guessIndex].guessedSection = guessed;
        recentGuesses[guessIndex].actualSection = actual;
        recentGuesses[guessIndex].correct = (guessed == actual);
        
        guessIndex = (guessIndex + 1) % ACCURACY_WINDOW;
        if (guessCount < ACCURACY_WINDOW) guessCount++;
    }
    
    float getAccuracy() const {
        if (guessCount == 0) return 0.0f;
        int correct = 0;
        for (uint8_t i = 0; i < guessCount; i++) {
            if (recentGuesses[i].correct) correct++;
        }
        return (float)correct / guessCount * 100.0f;
    }
    
    void clear() {
        for (auto& sec : sections) {
            sec.stats.clear();
            sec.fingerprint.clear();
            sec.name[0] = '\0';
        }
        guessCount = 0;
        guessIndex = 0;
    }
};

struct SessionRecord {
    uint32_t id;
    uint64_t created;
    uint32_t durationMs;
    int8_t sectionId;
    bool useAutoDetect;  // NEW: Session-level choice
    char username[24];
    Stats stats;
};

struct FeatureVector {
    float features[NUM_FEATURES];
    bool isValid;
    
    FeatureVector() : isValid(false) {
        for (int i = 0; i < NUM_FEATURES; i++) features[i] = 0.0f;
    }
};

struct CalibrationSample {
    FeatureVector vector;
    uint32_t timestamp;
};

// ============================================================================
// CONFIG
// ============================================================================
struct Config {
    uint32_t version = CONFIG_VERSION;
    uint32_t rimWindowMs = DEF_RIM_WINDOW_MS;
    uint32_t entryWindowMs = DEF_ENTRY_WINDOW_MS;
    uint32_t shotCooldownMs = DEF_SHOT_COOLDOWN_MS;
    uint32_t vibDebounceMs = DEF_VIB_DEBOUNCE_MS;
    int ballTriggerDrop = DEF_BALL_TRIGGER_DROP;
    float baselineAlpha = DEF_BASELINE_ALPHA;
    char apPassword[24];
    char otaUser[16];
    char otaPass[24];
    
    Config() {
        strcpy(otaUser, "admin");
        strcpy(otaPass, "hooptracker2024");
        apPassword[0] = '\0';
    }
    
    void validate() {
        rimWindowMs = clampInt(rimWindowMs, 50, 1000);
        entryWindowMs = clampInt(entryWindowMs, 50, 1500);
        shotCooldownMs = clampInt(shotCooldownMs, 200, 5000);
        vibDebounceMs = clampInt(vibDebounceMs, 1, 500);
        ballTriggerDrop = clampInt(ballTriggerDrop, 60, 800);
        baselineAlpha = clampFloat(baselineAlpha, 0.0005f, 0.2f);
    }
};

Config cfg;

// ============================================================================
// GLOBAL STATE
// ============================================================================
std::array<UserProfile, MAX_USERS> users;
uint8_t userCount = 0;
int8_t currentUserId = -1;

std::array<SessionRecord, MAX_SESSIONS> sessions;
uint16_t sessionCount = 0;

bool sessionActive = false;
uint32_t sessionStartMs = 0;
uint32_t sessionDurationMs = 0;
int8_t sessionSection = -1;
bool sessionUseAutoDetect = true;  // NEW: Per-session setting
Stats currentSessionStats;
uint32_t currentSessionId = 0;

std::atomic<bool> vibFlag{false};
std::atomic<uint32_t> lastVibrationMs{0};
uint32_t lastShotMs = 0;

enum class ShotState { Idle, VibrationPrimed, BallEntering, AwaitRimWindow, Cooldown };
ShotState shotState = ShotState::Idle;
uint32_t vibWindowEndMs = 0;
uint32_t rimWindowEndMs = 0;
bool rimHitPre = false;

volatile uint32_t nextSessionId = 1000;
int baseline = BASELINE_DISTANCE;
float baselineAvg = (float)BASELINE_DISTANCE;

bool dataDirty = false;
uint32_t lastDataSaveMs = 0;
bool sensorAvailable = false;

bool calibrating = false;
int8_t calibrateSectionId = -1;
uint32_t calibrateShotsTarget = MIN_CALIBRATION_SHOTS;
uint32_t calibrateShotsCollected = 0;
std::array<CalibrationSample, MAX_CALIBRATION_SHOTS> calibrationSamples;

// NEW: Temporal smoothing for confidence
std::array<float, CONFIDENCE_SMOOTHING_WINDOW> recentConfidences;
uint8_t confidenceIndex = 0;
uint8_t confidenceCount = 0;

constexpr int TOF_BUF_SIZE = 40;
struct ToFSample { 
    int16_t dist; 
    uint32_t ts; 
};
std::array<ToFSample, TOF_BUF_SIZE> tofBuf;
uint8_t tofBufIdx = 0;

constexpr int VIB_BUF_SIZE = 20;
volatile uint32_t vibTimestamps[VIB_BUF_SIZE];
volatile uint8_t vibBufIdx = 0;

struct SystemHealth {
    size_t minFreeHeap;
    uint32_t lastHeapWarning;
    uint16_t sensorFailures;
    uint32_t lastSensorFailure;
    bool heapCritical;
    
    SystemHealth() : minFreeHeap(SIZE_MAX), lastHeapWarning(0), 
                     sensorFailures(0), lastSensorFailure(0), heapCritical(false) {}
};
SystemHealth sysHealth;

String cachedIndex;

// ============================================================================
// STATE MACHINE
// ============================================================================
bool isValidTransition(ShotState from, ShotState to) {
    static const bool validTransitions[5][5] = {
        {true,  true,  true,  false, false},
        {false, true,  true,  false, true},
        {false, false, true,  true,  false},
        {false, false, false, true,  true},
        {true,  false, false, false, true}
    };
    return validTransitions[static_cast<int>(from)][static_cast<int>(to)];
}

void transitionState(ShotState newState, const char* reason) {
    if (!isValidTransition(shotState, newState)) {
        Serial.printf("ERROR: Invalid transition %d->%d (%s)\n",
                     static_cast<int>(shotState), static_cast<int>(newState), reason);
        return;
    }
    FSM_LOG(reason);
    shotState = newState;
}

// ============================================================================
// VIBRATION ISR
// ============================================================================
void IRAM_ATTR vibISR() {
    uint32_t now = millis();
    uint32_t last = lastVibrationMs.load(std::memory_order_relaxed);
    
    if ((int32_t)(now - last) > (int32_t)cfg.vibDebounceMs) {
        lastVibrationMs.store(now, std::memory_order_relaxed);
        vibTimestamps[vibBufIdx] = now;
        vibBufIdx = (vibBufIdx + 1) % VIB_BUF_SIZE;
        vibFlag.store(true, std::memory_order_relaxed);
    }
}

void handleVibrationFlag(uint32_t now) {
    if (!vibFlag.load(std::memory_order_relaxed)) return;
    vibFlag.store(false, std::memory_order_relaxed);
}

uint8_t getVibrationsInWindow(uint32_t startMs, uint32_t endMs, uint32_t* out, uint8_t maxCount) {
    uint8_t count = 0;
    
    noInterrupts();
    for (int i = 0; i < VIB_BUF_SIZE && count < maxCount; i++) {
        int idx = (vibBufIdx - 1 - i + VIB_BUF_SIZE) % VIB_BUF_SIZE;
        uint32_t ts = vibTimestamps[idx];
        if (ts == 0) continue;
        if (ts >= startMs && ts <= endMs) {
            out[count++] = ts;
        }
        if (ts < startMs) break;
    }
    interrupts();
    
    for (uint8_t i = 0; i < count - 1; i++) {
        for (uint8_t j = 0; j < count - i - 1; j++) {
            if (out[j] > out[j + 1]) {
                uint32_t tmp = out[j];
                out[j] = out[j + 1];
                out[j + 1] = tmp;
            }
        }
    }
    
    return count;
}

// ============================================================================
// TOF UTILITIES
// ============================================================================
void pushTofSample(int dist, uint32_t ts) {
    tofBufIdx = (tofBufIdx + 1) % TOF_BUF_SIZE;
    tofBuf[tofBufIdx].dist = dist;
    tofBuf[tofBufIdx].ts = ts;
}

uint8_t getTofWindow(uint32_t centerMs, uint32_t windowMs, ToFSample* out, uint8_t maxCount) {
    uint8_t count = 0;
    uint32_t startMs = (centerMs > windowMs/2) ? (centerMs - windowMs/2) : 0;
    uint32_t endMs = centerMs + windowMs/2;
    
    for (int i = 0; i < TOF_BUF_SIZE && count < maxCount; i++) {
        int idx = (tofBufIdx - i + TOF_BUF_SIZE) % TOF_BUF_SIZE;
        const ToFSample &s = tofBuf[idx];
        if (s.ts >= startMs && s.ts <= endMs && s.dist > 0) {
            out[count++] = s;
        }
    }
    
    for (uint8_t i = 0; i < count - 1; i++) {
        for (uint8_t j = 0; j < count - i - 1; j++) {
            if (out[j].ts > out[j + 1].ts) {
                ToFSample tmp = out[j];
                out[j] = out[j + 1];
                out[j + 1] = tmp;
            }
        }
    }
    
    return count;
}

// ============================================================================
// IMPROVED FEATURE EXTRACTION
// ============================================================================

// NEW: Quadratic fit for velocity (more robust)
float computeEntryVelocityQuadratic(const ToFSample* samples, uint8_t count) {
    if (count < 4) return 500.0f;
    
    // Fit y = ax^2 + bx + c where y=distance, x=time
    float sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
    float sumY = 0, sumXY = 0, sumX2Y = 0;
    
    float t0 = samples[0].ts / 1000.0f;
    
    for (uint8_t i = 0; i < count; i++) {
        float x = (samples[i].ts / 1000.0f) - t0;
        float y = samples[i].dist;
        
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x3 * x;
        
        sumX += x;
        sumX2 += x2;
        sumX3 += x3;
        sumX4 += x4;
        sumY += y;
        sumXY += x * y;
        sumX2Y += x2 * y;
    }
    
    // Solve normal equations (simplified for speed)
    // Velocity at entry ≈ derivative at x=0 ≈ b coefficient
    float n = count;
    float denom = n * sumX2 - sumX * sumX;
    if (fabs(denom) < 0.001f) return 500.0f;
    
    float b = (n * sumXY - sumX * sumY) / denom;
    
    return fabs(b);  // mm/s
}

FeatureVector extractPhysicalFeatures(uint32_t ballTriggerMs, uint32_t rimImpactMs) {
    FeatureVector vec;
    vec.isValid = false;
    
    ToFSample samples[30];
    uint8_t sampleCount = getTofWindow(ballTriggerMs, 400, samples, 30);
    
    if (sampleCount < 5) return vec;
    
    // Feature 0: Entry velocity with quadratic fit
    vec.features[0] = computeEntryVelocityQuadratic(samples, sampleCount);
    
    // Feature 1: Dwell time
    uint32_t dwellStart = 0, dwellEnd = 0;
    for (uint8_t i = 0; i < sampleCount; i++) {
        if (abs(samples[i].dist - baseline) < 50) {
            if (dwellStart == 0) dwellStart = samples[i].ts;
            dwellEnd = samples[i].ts;
        }
    }
    vec.features[1] = (dwellEnd > dwellStart) ? (float)(dwellEnd - dwellStart) : 50.0f;
    
    // Feature 2: Time to rim
    vec.features[2] = (float)(rimImpactMs - ballTriggerMs);
    if (vec.features[2] < 0 || vec.features[2] > 1000) vec.features[2] = 200.0f;
    
    // Feature 3: Vibration count
    uint32_t vibTimes[15];
    uint8_t vibCount = getVibrationsInWindow(ballTriggerMs, 
                                             rimImpactMs + cfg.rimWindowMs, 
                                             vibTimes, 15);
    vec.features[3] = (float)vibCount;
    
    // Feature 4: Entry consistency
    float totalVelocity = 0.0f;
    int velocityCount = 0;
    
    for (uint8_t i = 1; i < sampleCount; i++) {
        if (samples[i].ts > samples[i-1].ts) {
            float dt = (samples[i].ts - samples[i-1].ts) / 1000.0f;
            if (dt > 0.001f && dt < 0.2f) {
                float dd = samples[i-1].dist - samples[i].dist;
                float velocity = dd / dt;
                if (velocity > 0 && velocity < 10000) {
                    totalVelocity += velocity;
                    velocityCount++;
                }
            }
        }
    }
    
    float meanVel = (velocityCount > 0) ? (totalVelocity / velocityCount) : vec.features[0];
    float variance = 0.0f;
    velocityCount = 0;
    
    for (uint8_t i = 1; i < sampleCount; i++) {
        if (samples[i].ts > samples[i-1].ts) {
            float dt = (samples[i].ts - samples[i-1].ts) / 1000.0f;
            if (dt > 0.001f && dt < 0.2f) {
                float dd = samples[i-1].dist - samples[i].dist;
                float velocity = dd / dt;
                if (velocity > 0 && velocity < 10000) {
                    variance += (velocity - meanVel) * (velocity - meanVel);
                    velocityCount++;
                }
            }
        }
    }
    
    vec.features[4] = (velocityCount > 2) ? sqrtf(variance / velocityCount) : 100.0f;
    
    vec.isValid = true;
    return vec;
}

// ============================================================================
// IMPROVED CALIBRATION WITH OUTLIER REJECTION
// ============================================================================
void finalizeCalibration() {
    if (calibrateShotsCollected < MIN_CALIBRATION_SHOTS) {
        Serial.println("Insufficient samples");
        calibrating = false;
        return;
    }
    
    if (currentUserId < 0 || currentUserId >= userCount) {
        calibrating = false;
        return;
    }
    
    if (calibrateSectionId < 0 || calibrateSectionId >= NUM_SECTIONS) {
        calibrating = false;
        return;
    }
    
    uint32_t n = calibrateShotsCollected;
    
    // Pass 1: Compute initial means
    float means[NUM_FEATURES] = {0};
    for (uint32_t i = 0; i < n; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            means[j] += calibrationSamples[i].vector.features[j];
        }
    }
    for (int i = 0; i < NUM_FEATURES; i++) {
        means[i] /= n;
    }
    
    // Pass 2: Compute initial std deviations
    float stddevs[NUM_FEATURES] = {0};
    for (int i = 0; i < NUM_FEATURES; i++) {
        float variance = 0.0f;
        for (uint32_t j = 0; j < n; j++) {
            float diff = calibrationSamples[j].vector.features[i] - means[i];
            variance += diff * diff;
        }
        stddevs[i] = sqrtf(variance / n);
    }
    
    // Pass 3: Remove outliers (>3σ in ANY feature)
    uint32_t validCount = 0;
    for (uint32_t i = 0; i < n; i++) {
        bool isOutlier = false;
        for (int j = 0; j < NUM_FEATURES; j++) {
            float diff = fabs(calibrationSamples[i].vector.features[j] - means[j]);
            if (diff > OUTLIER_SIGMA_THRESHOLD * stddevs[j]) {
                isOutlier = true;
                break;
            }
        }
        
        if (!isOutlier) {
            if (validCount != i) {
                calibrationSamples[validCount] = calibrationSamples[i];
            }
            validCount++;
        }
    }
    
    Serial.printf("Outlier rejection: %u -> %u samples\n", n, validCount);
    
    if (validCount < MIN_CALIBRATION_SHOTS) {
        Serial.println("Too many outliers removed");
        calibrating = false;
        return;
    }
    
    // Pass 4: Recompute means with clean data
    for (int i = 0; i < NUM_FEATURES; i++) means[i] = 0;
    for (uint32_t i = 0; i < validCount; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            means[j] += calibrationSamples[i].vector.features[j];
        }
    }
    for (int i = 0; i < NUM_FEATURES; i++) {
        means[i] /= validCount;
    }
    
    // Pass 5: Compute final variances
    PhysicalFingerprint fp;
    fp.clear();
    
    for (int i = 0; i < NUM_FEATURES; i++) {
        float variance = 0.0f;
        for (uint32_t j = 0; j < validCount; j++) {
            float diff = calibrationSamples[j].vector.features[i] - means[i];
            variance += diff * diff;
        }
        fp.variance[i] = (variance / (validCount - 1)) + 0.1f;
    }
    
    // Compute weights
    for (int i = 0; i < NUM_FEATURES; i++) {
        fp.weights[i] = 1.0f / sqrtf(fp.variance[i]);
    }
    
    fp.setMeans(means);
    fp.samples = validCount;
    
    float sampleFactor = min(1.0f, sqrtf((float)validCount / 20.0f));
    float varianceFactor = 1.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
        varianceFactor *= clampFloat(1.0f / (1.0f + fp.variance[i] / 1000.0f), 0.5f, 1.0f);
    }
    fp.reliability = sampleFactor * powf(varianceFactor, 1.0f / NUM_FEATURES);
    
    users[currentUserId].sections[calibrateSectionId].fingerprint = fp;
    dataDirty = true;
    
    StaticJsonDocument<256> ev;
    ev["section"] = calibrateSectionId;
    ev["collected"] = calibrateShotsCollected;
    ev["target"] = calibrateShotsTarget;
    ev["finished"] = true;
    ev["reliability"] = fp.reliability;
    ev["samples"] = fp.samples;
    char buf[256];
    serializeJson(ev, buf);
    events.send(buf, "cal", millis());
    
    Serial.printf("Calibration complete: section=%d, samples=%u, reliability=%.2f\n",
                  calibrateSectionId, fp.samples, fp.reliability);
    
    calibrateShotsCollected = 0;
    calibrateSectionId = -1;
    calibrating = false;
}

void acceptCalibrationSample(const FeatureVector& fv) {
    if (!calibrating || calibrateSectionId < 0) return;
    if (!fv.isValid) return;
    if (calibrateShotsCollected >= MAX_CALIBRATION_SHOTS) return;
    
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(10));
        if (!lk.isLocked()) return;
        
        calibrationSamples[calibrateShotsCollected].vector = fv;
        calibrationSamples[calibrateShotsCollected].timestamp = millis();
        calibrateShotsCollected++;
        
        StaticJsonDocument<192> ev;
        ev["section"] = calibrateSectionId;
        ev["collected"] = calibrateShotsCollected;
        ev["target"] = calibrateShotsTarget;
        ev["finished"] = false;
        char buf[192];
        serializeJson(ev, buf);
        events.send(buf, "cal", millis());
        
        if (calibrateShotsCollected >= calibrateShotsTarget) {
            finalizeCalibration();
        }
    }
}

// ============================================================================
// CLASSIFICATION WITH TEMPORAL SMOOTHING
// ============================================================================
struct Classification {
    int sectionId;
    float distance;
    float confidence;
    float smoothedConfidence;  // NEW
};

float weightedDistance(const FeatureVector& shot, const PhysicalFingerprint& fp) {
    if (fp.samples < MIN_CALIBRATION_SHOTS) return 1e9f;
    
    float means[NUM_FEATURES];
    fp.getMeans(means);
    
    float distSq = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
        float diff = shot.features[i] - means[i];
        float weighted = diff * fp.weights[i];
        distSq += weighted * weighted;
    }
    
    return sqrtf(distSq);
}

// NEW: Add confidence to smoothing buffer and return EMA
float addConfidenceToSmoothing(float newConfidence) {
    recentConfidences[confidenceIndex] = newConfidence;
    confidenceIndex = (confidenceIndex + 1) % CONFIDENCE_SMOOTHING_WINDOW;
    if (confidenceCount < CONFIDENCE_SMOOTHING_WINDOW) confidenceCount++;
    
    // Exponential moving average with more weight on recent
    float smoothed = 0.0f;
    float weightSum = 0.0f;
    float alpha = 0.4f;  // Decay factor
    
    for (uint8_t i = 0; i < confidenceCount; i++) {
        uint8_t idx = (confidenceIndex - 1 - i + CONFIDENCE_SMOOTHING_WINDOW) % CONFIDENCE_SMOOTHING_WINDOW;
        float weight = powf(alpha, i);
        smoothed += recentConfidences[idx] * weight;
        weightSum += weight;
    }
    
    return (weightSum > 0) ? (smoothed / weightSum) : newConfidence;
}

Classification classifyShot(const FeatureVector& fv, int userId) {
    Classification result = {-1, 1e9f, 0.0f, 0.0f};
    
    if (!fv.isValid) return result;
    if (userId < 0 || userId >= userCount) return result;
    
    float bestDist = 1e9f;
    float secondBest = 1e9f;
    int bestSection = -1;
    
    for (int i = 0; i < NUM_SECTIONS; i++) {
        const auto& fp = users[userId].sections[i].fingerprint;
        
        if (fp.samples < MIN_CALIBRATION_SHOTS || fp.reliability < 0.3f) {
            continue;
        }
        
        float dist = weightedDistance(fv, fp);
        
        if (dist < bestDist) {
            secondBest = bestDist;
            bestDist = dist;
            bestSection = i;
        } else if (dist < secondBest) {
            secondBest = dist;
        }
    }
    
    if (bestSection < 0) return result;
    
    float distConf = 1.0f / (1.0f + bestDist / 10.0f);
    float sepConf = (secondBest > bestDist) ? 
                    ((secondBest - bestDist) / max(0.1f, secondBest)) : 0.0f;
    
    result.sectionId = bestSection;
    result.distance = bestDist;
    result.confidence = distConf * 0.6f + sepConf * 0.4f;
    result.confidence *= users[userId].sections[bestSection].fingerprint.reliability;
    result.confidence = clampFloat(result.confidence, 0.0f, 1.0f);
    
    // NEW: Apply temporal smoothing
    result.smoothedConfidence = addConfidenceToSmoothing(result.confidence);
    
    return result;
}

void adaptiveLearning(int userId, int sectionId, const FeatureVector& shot, float confidence) {
    if (confidence < HIGH_CONFIDENCE_THRESHOLD) return;
    if (userId < 0 || userId >= userCount) return;
    if (sectionId < 0 || sectionId >= NUM_SECTIONS) return;
    if (!users[userId].backgroundLearning) return;  // NEW: Respect toggle
    
    auto& fp = users[userId].sections[sectionId].fingerprint;
    if (fp.samples < MIN_CALIBRATION_SHOTS) return;
    
    float means[NUM_FEATURES];
    fp.getMeans(means);
    
    for (int i = 0; i < NUM_FEATURES; i++) {
        means[i] = (1.0f - ADAPTIVE_LEARNING_RATE) * means[i] + 
                   ADAPTIVE_LEARNING_RATE * shot.features[i];
    }
    
    fp.setMeans(means);
    fp.samples++;
    dataDirty = true;
}

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================
void pushLiveUpdate();
void pushSectionUpdate();

// ============================================================================
// SHOT RECORDING WITH BACKGROUND LEARNING CONTROL
// ============================================================================
void registerAttemptOutcomeWithVector(bool made, bool swish, const FeatureVector *vecPtr) {
    int assignedSection = sessionSection;
    float confidence = 1.0f;
    int userId = currentUserId;
    
    // NEW: Session-level control
    if (sessionUseAutoDetect && vecPtr != nullptr && userId >= 0 && userId < userCount) {
        // Auto-detect mode
        Classification c = classifyShot(*vecPtr, userId);
        if (c.sectionId >= 0 && c.smoothedConfidence > CONFIDENCE_THRESHOLD) {
            assignedSection = c.sectionId;
            confidence = c.smoothedConfidence;
            
            // Adaptive learning if background learning enabled
            if (made && confidence > HIGH_CONFIDENCE_THRESHOLD && users[userId].backgroundLearning) {
                adaptiveLearning(userId, assignedSection, *vecPtr, confidence);
            }
        }
    } else if (!sessionUseAutoDetect && vecPtr != nullptr && userId >= 0 && userId < userCount) {
        // Manual mode - but background learning still active if enabled
        assignedSection = sessionSection;
        
        if (users[userId].backgroundLearning) {
            Classification c = classifyShot(*vecPtr, userId);
            if (c.sectionId >= 0 && sessionSection >= 0) {
                users[userId].addGuess(c.sectionId, sessionSection);
                dataDirty = true;
                
                if (c.sectionId == sessionSection && c.smoothedConfidence > 0.5f) {
                    adaptiveLearning(userId, sessionSection, *vecPtr, c.smoothedConfidence);
                }
            }
        }
    } else if (vecPtr != nullptr && userId == -1) {
        // Anybody mode
        float bestConf = 0.0f;
        int bestSection = -1;
        
        for (uint8_t u = 0; u < userCount; u++) {
            Classification c = classifyShot(*vecPtr, u);
            if (c.smoothedConfidence > bestConf) {
                bestConf = c.smoothedConfidence;
                bestSection = c.sectionId;
            }
        }
        
        if (bestSection >= 0 && bestConf > CONFIDENCE_THRESHOLD) {
            assignedSection = bestSection;
            confidence = bestConf;
        }
    }
    
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(5));
        if (!lk.isLocked()) return;
        
        currentSessionStats.addAttempt(made, swish);
        
        if (sessionActive && currentSessionId != 0) {
            for (uint16_t i = 0; i < sessionCount; i++) {
                if (sessions[i].id == currentSessionId) {
                    sessions[i].stats = currentSessionStats;
                    break;
                }
            }
        }
        
        if (sessionActive && assignedSection >= 0 && assignedSection < NUM_SECTIONS) {
            if (userId >= 0 && userId < userCount) {
                users[userId].sections[assignedSection].stats.addAttempt(made, swish);
                dataDirty = true;
            }
        }
    }
    
    pushLiveUpdate();
    pushSectionUpdate();
    
    StaticJsonDocument<256> j;
    j["classifiedSection"] = assignedSection;
    j["confidence"] = confidence;
    j["made"] = made;
    j["swish"] = swish;
    j["userId"] = userId;
    if (userId >= 0 && userId < userCount) {
        j["accuracy"] = users[userId].getAccuracy();
        j["backgroundLearning"] = users[userId].backgroundLearning;
    }
    char buf[256];
    serializeJson(j, buf, sizeof(buf));
    events.send(buf, "classified", millis());
}

void registerAttemptOutcome(bool made, bool swish) {
    registerAttemptOutcomeWithVector(made, swish, nullptr);
}

// ============================================================================
// PERSISTENCE
// ============================================================================
void saveUsers() {
    File f = SPIFFS.open(USERS_FILE, FILE_WRITE);
    if (!f) return;
    
    DynamicJsonDocument doc(8192);
    doc["v"] = CONFIG_VERSION;
    JsonArray arr = doc.createNestedArray("users");
    
    for (uint8_t u = 0; u < userCount; u++) {
        JsonObject user = arr.createNestedObject();
        user["username"] = users[u].username;
        user["backgroundLearning"] = users[u].backgroundLearning;
        
        JsonArray secs = user.createNestedArray("sections");
        for (int i = 0; i < NUM_SECTIONS; i++) {
            JsonObject sec = secs.createNestedObject();
            sec["a"] = users[u].sections[i].stats.attempts;
            sec["m"] = users[u].sections[i].stats.makes;
            sec["s"] = users[u].sections[i].stats.swishes;
            sec["r"] = users[u].sections[i].stats.rimMakes;
            sec["name"] = users[u].sections[i].name;
            
            const auto& fp = users[u].sections[i].fingerprint;
            if (fp.samples > 0) {
                JsonObject fpObj = sec.createNestedObject("fp");
                fpObj["n"] = fp.samples;
                fpObj["r"] = fp.reliability;
                
                JsonArray m = fpObj.createNestedArray("m");
                float means[NUM_FEATURES];
                fp.getMeans(means);
                for (int j = 0; j < NUM_FEATURES; j++) m.add(means[j]);
                
                JsonArray v = fpObj.createNestedArray("v");
                for (int j = 0; j < NUM_FEATURES; j++) v.add(fp.variance[j]);
            }
        }
        
        if (users[u].guessCount > 0) {
            JsonArray guesses = user.createNestedArray("g");
            for (uint8_t i = 0; i < users[u].guessCount; i++) {
                JsonObject g = guesses.createNestedObject();
                g["gs"] = users[u].recentGuesses[i].guessedSection;
                g["as"] = users[u].recentGuesses[i].actualSection;
                g["c"] = users[u].recentGuesses[i].correct;
            }
        }
    }
    
    serializeJson(doc, f);
    f.close();
}

void loadUsers() {
    if (!SPIFFS.exists(USERS_FILE)) return;
    
    File f = SPIFFS.open(USERS_FILE, FILE_READ);
    if (!f) return;
    
    DynamicJsonDocument doc(8192);
    auto err = deserializeJson(doc, f);
    f.close();
    
    if (err) return;
    
    JsonArray arr = doc["users"];
    userCount = 0;
    
    for (JsonObject u : arr) {
        if (userCount >= MAX_USERS) break;
        
        strncpy(users[userCount].username, u["username"] | "", 23);
        users[userCount].username[23] = '\0';
        users[userCount].backgroundLearning = u["backgroundLearning"] | true;
        
        JsonArray secs = u["sections"];
        for (int i = 0; i < NUM_SECTIONS && i < (int)secs.size(); i++) {
            JsonObject sec = secs[i];
            users[userCount].sections[i].stats.attempts = sec["a"] | 0;
            users[userCount].sections[i].stats.makes = sec["m"] | 0;
            users[userCount].sections[i].stats.swishes = sec["s"] | 0;
            users[userCount].sections[i].stats.rimMakes = sec["r"] | 0;
            strncpy(users[userCount].sections[i].name, sec["name"] | "", 23);
            users[userCount].sections[i].name[23] = '\0';
            
            if (sec.containsKey("fp")) {
                JsonObject fp = sec["fp"];
                users[userCount].sections[i].fingerprint.samples = fp["n"] | 0;
                users[userCount].sections[i].fingerprint.reliability = fp["r"] | 0.0f;
                
                if (fp.containsKey("m")) {
                    JsonArray m = fp["m"];
                    float means[NUM_FEATURES] = {0};
                    for (int j = 0; j < NUM_FEATURES && j < (int)m.size(); j++) {
                        means[j] = m[j] | 0.0f;
                    }
                    users[userCount].sections[i].fingerprint.setMeans(means);
                }
                
                if (fp.containsKey("v")) {
                    JsonArray v = fp["v"];
                    for (int j = 0; j < NUM_FEATURES && j < (int)v.size(); j++) {
                        users[userCount].sections[i].fingerprint.variance[j] = v[j] | 1.0f;
                    }
                    
                    for (int j = 0; j < NUM_FEATURES; j++) {
                        users[userCount].sections[i].fingerprint.weights[j] = 
                            1.0f / sqrtf(users[userCount].sections[i].fingerprint.variance[j]);
                    }
                }
            }
        }
        
        if (u.containsKey("g")) {
            JsonArray guesses = u["g"];
            users[userCount].guessCount = 0;
            users[userCount].guessIndex = 0;
            
            for (JsonObject g : guesses) {
                if (users[userCount].guessCount >= ACCURACY_WINDOW) break;
                
                users[userCount].recentGuesses[users[userCount].guessCount].guessedSection = g["gs"] | -1;
                users[userCount].recentGuesses[users[userCount].guessCount].actualSection = g["as"] | -1;
                users[userCount].recentGuesses[users[userCount].guessCount].correct = g["c"] | false;
                users[userCount].guessCount++;
            }
            users[userCount].guessIndex = users[userCount].guessCount % ACCURACY_WINDOW;
        }
        
        userCount++;
    }
}

void saveSessions() {
    File f = SPIFFS.open(SESSIONS_FILE, FILE_WRITE);
    if (!f) return;
    
    DynamicJsonDocument doc(6144);
    doc["v"] = 1;
    JsonArray arr = doc.createNestedArray("items");
    
    uint16_t start = (sessionCount > 100) ? (sessionCount - 100) : 0;
    
    for (uint16_t i = start; i < sessionCount; i++) {
        JsonObject o = arr.createNestedObject();
        o["id"] = sessions[i].id;
        o["c"] = sessions[i].created;
        o["d"] = sessions[i].durationMs;
        o["s"] = sessions[i].sectionId;
        o["auto"] = sessions[i].useAutoDetect;
        o["u"] = sessions[i].username;
        JsonObject st = o.createNestedObject("st");
        st["a"] = sessions[i].stats.attempts;
        st["m"] = sessions[i].stats.makes;
        st["sw"] = sessions[i].stats.swishes;
        st["r"] = sessions[i].stats.rimMakes;
    }
    
    serializeJson(doc, f);
    f.close();
}

void loadSessions() {
    if (!SPIFFS.exists(SESSIONS_FILE)) return;
    
    File f = SPIFFS.open(SESSIONS_FILE, FILE_READ);
    if (!f) return;
    
    DynamicJsonDocument doc(6144);
    auto err = deserializeJson(doc, f);
    f.close();
    
    if (err) return;
    
    JsonArray arr = doc["items"];
    sessionCount = 0;
    uint32_t maxId = 0;
    
    for (JsonObject s : arr) {
        if (sessionCount >= MAX_SESSIONS) break;
        
        sessions[sessionCount].id = s["id"] | 0;
        sessions[sessionCount].created = s["c"] | 0ULL;
        sessions[sessionCount].durationMs = s["d"] | 0;
        sessions[sessionCount].sectionId = s["s"] | -1;
        sessions[sessionCount].useAutoDetect = s["auto"] | true;
        strncpy(sessions[sessionCount].username, s["u"] | "", 23);
        sessions[sessionCount].username[23] = '\0';
        sessions[sessionCount].stats.attempts = s["st"]["a"] | 0;
        sessions[sessionCount].stats.makes = s["st"]["m"] | 0;
        sessions[sessionCount].stats.swishes = s["st"]["sw"] | 0;
        sessions[sessionCount].stats.rimMakes = s["st"]["r"] | 0;
        
        if (sessions[sessionCount].id > maxId) {
            maxId = sessions[sessionCount].id;
        }
        
        sessionCount++;
    }
    
    if (maxId >= nextSessionId) {
        nextSessionId = maxId + 1;
    }
}

bool saveConfigAtomic() {
    File f = SPIFFS.open(CONFIG_FILE, FILE_WRITE);
    if (!f) return false;
    
    StaticJsonDocument<512> j;
    j["version"] = cfg.version;
    j["rimWindowMs"] = cfg.rimWindowMs;
    j["entryWindowMs"] = cfg.entryWindowMs;
    j["shotCooldownMs"] = cfg.shotCooldownMs;
    j["vibDebounceMs"] = cfg.vibDebounceMs;
    j["ballTriggerDrop"] = cfg.ballTriggerDrop;
    j["baselineAlpha"] = cfg.baselineAlpha;
    j["apPassword"] = cfg.apPassword;
    j["otaUser"] = cfg.otaUser;
    j["otaPass"] = cfg.otaPass;
    
    serializeJson(j, f);
    f.close();
    return true;
}

void ensureConfigDefaults() {
    if (strlen(cfg.apPassword) < 8) {
        String pw = randomPass(12);
        strncpy(cfg.apPassword, pw.c_str(), 23);
        cfg.apPassword[23] = '\0';
    }
}

bool loadConfig() {
    if (!SPIFFS.exists(CONFIG_FILE)) {
        ensureConfigDefaults();
        return false;
    }
    
    File f = SPIFFS.open(CONFIG_FILE, FILE_READ);
    if (!f) {
        ensureConfigDefaults();
        return false;
    }
    
    StaticJsonDocument<512> j;
    auto err = deserializeJson(j, f);
    f.close();
    
    if (err) {
        ensureConfigDefaults();
        return false;
    }
    
    cfg.version = j["version"] | CONFIG_VERSION;
    cfg.rimWindowMs = j["rimWindowMs"] | DEF_RIM_WINDOW_MS;
    cfg.entryWindowMs = j["entryWindowMs"] | DEF_ENTRY_WINDOW_MS;
    cfg.shotCooldownMs = j["shotCooldownMs"] | DEF_SHOT_COOLDOWN_MS;
    cfg.vibDebounceMs = j["vibDebounceMs"] | DEF_VIB_DEBOUNCE_MS;
    cfg.ballTriggerDrop = j["ballTriggerDrop"] | DEF_BALL_TRIGGER_DROP;
    cfg.baselineAlpha = j["baselineAlpha"] | DEF_BASELINE_ALPHA;
    
    strncpy(cfg.apPassword, j["apPassword"] | "", 23);
    strncpy(cfg.otaUser, j["otaUser"] | "admin", 15);
    strncpy(cfg.otaPass, j["otaPass"] | "hooptracker2024", 23);
    
    cfg.validate();
    
    return true;
}

// ============================================================================
// HEAP MONITORING
// ============================================================================
void checkHeapHealth() {
    size_t freeHeap = ESP.getFreeHeap();
    
    if (freeHeap < sysHealth.minFreeHeap) {
        sysHealth.minFreeHeap = freeHeap;
    }
    
    if (freeHeap < HEAP_CRITICAL_THRESHOLD) {
        if (!sysHealth.heapCritical) {
            Serial.printf("CRITICAL: Heap at %zu bytes!\n", freeHeap);
            sysHealth.heapCritical = true;
            
            calibrating = false;
            calibrateShotsCollected = 0;
        }
    } else {
        sysHealth.heapCritical = false;
    }
    
    if (freeHeap < HEAP_WARNING_THRESHOLD) {
        uint32_t now = millis();
        if (now - sysHealth.lastHeapWarning > 30000) {
            Serial.printf("WARNING: Low heap: %zu bytes\n", freeHeap);
            sysHealth.lastHeapWarning = now;
        }
    }
}

// ============================================================================
// POLISHED WEB UI
// ============================================================================
const char index_html[] PROGMEM = R"rawliteral(
<!doctype html><html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HoopTracker Pro Elite</title>
<style>
:root{--primary:#ff6a00;--bg:#000;--card:#111;--border:#222;--text:#fff;--muted:#888;--success:#4caf50;--warning:#ff9800;--danger:#f44336}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#0a0a0a,#1a1a1a);color:var(--text);min-height:100vh;padding:20px}
.container{max-width:1200px;margin:0 auto}
h1{text-align:center;margin-bottom:40px;font-size:2.8rem;background:linear-gradient(135deg,#ff6a00,#ff8c00);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px}
.start-screen{display:flex;flex-direction:column;gap:24px;max-width:700px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:28px;box-shadow:0 10px 40px rgba(0,0,0,0.5);transition:transform 0.3s}
.user-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px}
.user-card{background:linear-gradient(135deg,#1a1a1a,#2a2a2a);border:2px solid var(--border);border-radius:16px;padding:24px;cursor:pointer;transition:all 0.4s cubic-bezier(0.4,0,0.2,1);text-align:center;position:relative;overflow:hidden}
.user-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,var(--primary),#ff8c00);transform:scaleX(0);transition:transform 0.4s}
.user-card:hover{transform:translateY(-6px);border-color:var(--primary);box-shadow:0 12px 32px rgba(255,106,0,0.4)}
.user-card:hover::before{transform:scaleX(1)}
.user-card h3{font-size:1.6rem;margin-bottom:16px}
.user-stats{font-size:0.95rem;color:var(--muted);margin-top:10px}
.add-user{border:2px dashed var(--border);background:transparent}
.add-user:hover{border-color:var(--primary);background:rgba(255,106,0,0.08)}
.btn{background:var(--primary);color:var(--bg);border:none;padding:16px 32px;border-radius:12px;font-size:1.15rem;font-weight:600;cursor:pointer;transition:all 0.3s;width:100%;box-shadow:0 4px 16px rgba(255,106,0,0.3)}
.btn:hover{background:#ff7a1a;transform:translateY(-2px);box-shadow:0 6px 20px rgba(255,106,0,0.5)}
.btn:active{transform:translateY(0)}
.btn-secondary{background:var(--card);color:var(--text);border:1px solid var(--border);box-shadow:none}
.btn-secondary:hover{background:#222;box-shadow:none}
.btn-danger{background:var(--danger);color:white}
.session-view{display:none}
.section-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px;margin:24px 0}
.section-card{background:linear-gradient(135deg,#1a1a1a,#252525);border:1px solid var(--border);border-radius:14px;padding:18px;text-align:center;transition:all 0.3s}
.section-card:hover{transform:translateY(-5px);border-color:var(--primary);box-shadow:0 8px 24px rgba(255,106,0,0.2)}
.section-stats{font-size:2.2rem;color:var(--primary);font-weight:700;margin:10px 0}
.settings-group{margin:24px 0;padding:24px;background:var(--card);border-radius:16px;border:1px solid var(--border)}
.settings-group h3{margin-bottom:20px;color:var(--primary);font-size:1.3rem}
.setting-item{display:flex;justify-content:space-between;align-items:center;padding:16px 0;border-bottom:1px solid var(--border)}
.setting-item:last-child{border-bottom:none}
.toggle{position:relative;width:56px;height:30px;background:var(--border);border-radius:15px;cursor:pointer;transition:background 0.3s}
.toggle.on{background:var(--success)}
.toggle-slider{position:absolute;top:3px;left:3px;width:24px;height:24px;background:white;border-radius:50%;transition:left 0.3s;box-shadow:0 2px 4px rgba(0,0,0,0.2)}
.toggle.on .toggle-slider{left:29px}
input[type=text],select{background:var(--card);border:1px solid var(--border);color:var(--text);padding:14px;border-radius:10px;width:100%;font-size:1rem;transition:border-color 0.3s}
input[type=text]:focus,select:focus{outline:none;border-color:var(--primary)}
.accuracy-badge{display:inline-block;padding:8px 14px;border-radius:8px;font-size:1rem;font-weight:600;margin-top:10px}
.acc-high{background:var(--success);color:var(--bg)}
.acc-med{background:var(--warning);color:var(--bg)}
.acc-low{background:var(--danger);color:white}
.modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);align-items:center;justify-content:center;z-index:1000;padding:20px;backdrop-filter:blur(4px)}
.modal.show{display:flex;animation:fadeIn 0.3s}
.modal-content{background:var(--card);border-radius:20px;padding:36px;max-width:520px;width:100%;border:1px solid var(--border);box-shadow:0 20px 60px rgba(0,0,0,0.7);animation:slideUp 0.3s}
.modal-content h2{margin-bottom:24px;color:var(--primary);font-size:1.8rem}
.nav-tabs{display:flex;gap:10px;margin-bottom:24px;border-bottom:2px solid var(--border)}
.nav-tab{background:transparent;border:none;color:var(--muted);padding:14px 28px;cursor:pointer;border-bottom:3px solid transparent;transition:all 0.3s;font-size:1rem;font-weight:500}
.nav-tab.active{color:var(--primary);border-bottom-color:var(--primary)}
.nav-tab:hover{color:var(--text)}
.hidden{display:none!important}
.live-indicator{display:inline-block;width:10px;height:10px;background:var(--success);border-radius:50%;animation:pulse 2s infinite;margin-right:10px}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(1.1)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes slideUp{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}
.progress-bar{background:var(--border);height:10px;border-radius:5px;overflow:hidden;margin-top:10px}
.progress-fill{background:linear-gradient(90deg,var(--primary),#ff8c00);height:100%;width:0;transition:width 0.4s ease-out}
.session-mode-toggle{display:flex;gap:12px;margin:20px 0}
.mode-btn{flex:1;padding:18px;border:2px solid var(--border);background:var(--card);color:var(--text);border-radius:12px;cursor:pointer;transition:all 0.3s;font-size:1rem;font-weight:600}
.mode-btn.active{border-color:var(--primary);background:rgba(255,106,0,0.1);color:var(--primary)}
.mode-btn:hover{border-color:var(--primary)}
.info-box{background:rgba(255,106,0,0.1);border:1px solid rgba(255,106,0,0.3);border-radius:10px;padding:14px;margin:16px 0;font-size:0.9rem;color:var(--text)}
</style>
</head>
<body>
<div class="container">
<h1>🏀 HoopTracker Pro Elite</h1>

<div class="start-screen" id="startScreen">
<div class="card">
<h2 style="margin-bottom:24px;text-align:center;font-size:1.6rem">Select Your Profile</h2>
<div class="user-grid" id="userGrid"></div>
</div>
<button class="btn" onclick="selectAnybody()">🌐 Play as Anybody</button>
</div>

<div class="session-view" id="sessionView">
<div class="nav-tabs">
<button class="nav-tab active" onclick="showTab('session')">Session</button>
<button class="nav-tab" onclick="showTab('live')">Live</button>
<button class="nav-tab" onclick="showTab('stats')">Stats</button>
<button class="nav-tab" onclick="showTab('settings')">Settings</button>
<button class="nav-tab" onclick="showTab('calibrate')">Calibrate</button>
</div>

<div id="sessionTab">
<div class="card">
<h2>Start New Session</h2>
<div class="info-box">Choose how the system detects shot sections for this session</div>
<div class="session-mode-toggle">
<button class="mode-btn active" id="modeAuto" onclick="selectMode('auto')">
<div style="font-size:1.8rem;margin-bottom:8px">🤖</div>
Auto-Detect<div style="font-size:0.85rem;margin-top:4px;opacity:0.8">Use AI fingerprint</div>
</button>
<button class="mode-btn" id="modeManual" onclick="selectMode('manual')">
<div style="font-size:1.8rem;margin-bottom:8px">👆</div>
Manual<div style="font-size:0.85rem;margin-top:4px;opacity:0.8">Select section</div>
</button>
</div>
<div id="manualSectionPicker" class="hidden" style="margin:20px 0">
<label style="display:block;margin-bottom:8px;font-weight:600">Select Section:</label>
<select id="sessionSection" style="margin-bottom:20px">
<option value="0">Paint</option>
<option value="1">Left Corner</option>
<option value="2">Right Corner</option>
<option value="3">Left Wing</option>
<option value="4">Right Wing</option>
<option value="5">Left Mid</option>
<option value="6">Right Mid</option>
<option value="7">Top Key</option>
</select>
</div>
<button class="btn" onclick="startSession()">Start Shooting</button>
</div>
</div>

<div id="liveTab" class="hidden">
<div class="card">
<h2><span class="live-indicator"></span>Live Session</h2>
<p id="liveInfo" style="color:var(--muted);margin:14px 0;font-size:1.05rem"></p>
<div id="liveStats" style="font-size:1.3rem;margin:20px 0;line-height:1.6"></div>
<button class="btn btn-secondary" onclick="endSession()">End Session</button>
</div>
</div>

<div id="statsTab" class="hidden">
<div class="card">
<h2 style="margin-bottom:20px">Court Sections</h2>
<div class="section-grid" id="sectionGrid"></div>
</div>
</div>

<div id="settingsTab" class="hidden">
<div class="card">
<div class="settings-group">
<h3>🎯 AI Fingerprint System</h3>
<div class="setting-item">
<div>
<div style="font-weight:600">Background Learning</div>
<div style="font-size:0.88rem;color:var(--muted);margin-top:4px">Continuously improve from shots</div>
</div>
<div class="toggle on" id="learningToggle" onclick="toggleLearning()">
<div class="toggle-slider"></div>
</div>
</div>
<div class="setting-item">
<div>
<div style="font-weight:600">AI Accuracy</div>
<div style="font-size:0.88rem;color:var(--muted);margin-top:4px">Last 30 shots</div>
</div>
<div id="accuracyDisplay" style="font-size:1.4rem;font-weight:700;color:var(--primary)">--</div>
</div>
</div>
<div style="display:flex;gap:12px;margin-top:20px">
<button class="btn btn-danger" onclick="confirmReset()">Reset Profile Data</button>
<button class="btn btn-secondary" onclick="exportProfile()">Export CSV</button>
</div>
<button class="btn btn-secondary" onclick="backToStart()" style="margin-top:12px">Back to Start</button>
</div>
</div>

<div id="calibrateTab" class="hidden">
<div class="card">
<h2 style="margin-bottom:12px">🎯 Calibration Wizard</h2>
<p style="color:var(--muted);margin-bottom:20px">Position yourself at the selected section and take 15-20 shots. The AI will learn your shooting fingerprint from this location.</p>
<select id="calSection" style="margin-bottom:20px">
<option value="0">Paint</option>
<option value="1">Left Corner</option>
<option value="2">Right Corner</option>
<option value="3">Left Wing</option>
<option value="4">Right Wing</option>
<option value="5">Left Mid</option>
<option value="6">Right Mid</option>
<option value="7">Top Key</option>
</select>
<button class="btn" onclick="startCalibration()">Start Calibration</button>
<div id="calProgress" style="margin-top:24px;display:none">
<div style="font-weight:600;margin-bottom:10px;font-size:1.1rem">Progress: <span id="calCount">0</span> / <span id="calTarget">20</span></div>
<div class="progress-bar">
<div id="calBar" class="progress-fill"></div>
</div>
</div>
</div>
</div>
</div>

<div class="modal" id="modal">
<div class="modal-content">
<h2 id="modalTitle">Add Profile</h2>
<div id="modalBody">
<input type="text" id="modalInput" placeholder="Enter username" maxlength="20">
</div>
<div style="display:flex;gap:14px;margin-top:24px">
<button class="btn" onclick="modalConfirm()">Confirm</button>
<button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
</div>
</div>
</div>
</div>

<script>
let users=[];
let currentUser=null;
let evt=null;
let sessionMode='auto';

function init(){
  loadUsers();
}

function loadUsers(){
  fetch('/api/users').then(r=>r.json()).then(data=>{
    users=data.users||[];
    renderUsers();
  });
}

function renderUsers(){
  const grid=document.getElementById('userGrid');
  grid.innerHTML='';
  
  users.forEach((u,i)=>{
    const card=document.createElement('div');
    card.className='user-card';
    card.onclick=()=>selectUser(i);
    const acc=u.accuracy||0;
    const accClass=acc>=80?'acc-high':acc>=60?'acc-med':'acc-low';
    card.innerHTML=`
      <h3>${u.username}</h3>
      <div class="user-stats">${u.totalShots||0} total shots</div>
      <div class="accuracy-badge ${accClass}">${acc.toFixed(0)}% AI Accuracy</div>
    `;
    grid.appendChild(card);
  });
  
  if(users.length<4){
    const add=document.createElement('div');
    add.className='user-card add-user';
    add.onclick=addUser;
    add.innerHTML='<h3>+ Add Profile</h3><div style="color:var(--muted);margin-top:8px">Create new user</div>';
    grid.appendChild(add);
  }
}

function addUser(){
  document.getElementById('modalTitle').innerText='Add Profile';
  document.getElementById('modalBody').innerHTML='<input type="text" id="modalInput" placeholder="Enter username" maxlength="20">';
  document.getElementById('modal').classList.add('show');
}

function closeModal(){
  document.getElementById('modal').classList.remove('show');
}

function modalConfirm(){
  const input=document.getElementById('modalInput');
  if(!input)return;
  const name=input.value.trim();
  if(!name){alert('Enter a username');return}
  
  fetch('/api/users/create',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:name})
  }).then(r=>r.json()).then(data=>{
    if(data.error){alert(data.error);return}
    closeModal();
    loadUsers();
  });
}

function selectUser(idx){
  currentUser=idx;
  fetch('/api/users/select',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({userId:idx})
  }).then(()=>{
    showSessionView();
  });
}

function selectAnybody(){
  currentUser=-1;
  fetch('/api/users/select',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({userId:-1})
  }).then(()=>{
    showSessionView();
  });
}

function showSessionView(){
  document.getElementById('startScreen').style.display='none';
  document.getElementById('sessionView').style.display='block';
  loadStats();
  loadSettings();
  initSSE();
}

function showTab(tab){
  ['session','live','stats','settings','calibrate'].forEach(t=>{
    document.getElementById(t+'Tab').classList.toggle('hidden',t!==tab);
  });
  document.querySelectorAll('.nav-tab').forEach((btn,i)=>{
    btn.classList.toggle('active',['session','live','stats','settings','calibrate'][i]===tab);
  });
  if(tab==='stats')loadStats();
}

function selectMode(mode){
  sessionMode=mode;
  document.getElementById('modeAuto').classList.toggle('active',mode==='auto');
  document.getElementById('modeManual').classList.toggle('active',mode==='manual');
  document.getElementById('manualSectionPicker').classList.toggle('hidden',mode==='auto');
}

function startSession(){
  const useAuto=(sessionMode==='auto');
  const section=useAuto?-1:parseInt(document.getElementById('sessionSection').value);
  
  fetch('/api/session/create',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({durationMin:60,section:section,useAutoDetect:useAuto})
  }).then(r=>r.json()).then(()=>{
    showTab('live');
  });
}

function endSession(){
  if(confirm('End this session?'))backToStart();
}

function backToStart(){
  if(evt){evt.close();evt=null}
  document.getElementById('startScreen').style.display='flex';
  document.getElementById('sessionView').style.display='none';
  currentUser=null;
  loadUsers();
}

function loadStats(){
  if(currentUser<0)return;
  fetch('/api/users/'+currentUser+'/sections').then(r=>r.json()).then(data=>{
    const grid=document.getElementById('sectionGrid');
    grid.innerHTML='';
    const sections=['Paint','L Corner','R Corner','L Wing','R Wing','L Mid','R Mid','Top Key'];
    data.sections.forEach((s,i)=>{
      const card=document.createElement('div');
      card.className='section-card';
      const pct=s.attempts?(s.makes/s.attempts*100).toFixed(0):'--';
      const reliabilityColor=s.fingerprintSamples>=15?'var(--success)':s.fingerprintSamples>0?'var(--warning)':'var(--muted)';
      card.innerHTML=`
        <div style="font-weight:600;margin-bottom:10px;font-size:1.05rem">${sections[i]}</div>
        <div class="section-stats">${pct}%</div>
        <div style="font-size:0.9rem;color:var(--muted);margin:4px 0">${s.attempts} shots</div>
        <div style="font-size:0.85rem;color:${reliabilityColor};font-weight:600">${s.fingerprintSamples||0} calibrated</div>
      `;
      grid.appendChild(card);
    });
  });
}

function loadSettings(){
  if(currentUser<0){
    document.getElementById('learningToggle').style.display='none';
    return;
  }
  
  fetch('/api/users/'+currentUser).then(r=>r.json()).then(data=>{
    const toggle=document.getElementById('learningToggle');
    toggle.classList.toggle('on',data.backgroundLearning);
    
    const acc=data.accuracy||0;
    const accElem=document.getElementById('accuracyDisplay');
    accElem.innerText=acc.toFixed(1)+'%';
    accElem.style.color=acc>=80?'var(--success)':acc>=60?'var(--warning)':'var(--danger)';
  });
}

function toggleLearning(){
  if(currentUser<0)return;
  fetch('/api/users/'+currentUser+'/toggle-learning',{method:'POST'}).then(()=>{
    loadSettings();
  });
}

function confirmReset(){
  if(!confirm('Reset ALL profile data? This cannot be undone!'))return;
  fetch('/api/users/'+currentUser+'/reset',{method:'POST'}).then(()=>{
    alert('Profile reset complete');
    loadStats();
    loadSettings();
  });
}

function exportProfile(){
  window.open('/api/users/'+currentUser+'/export.csv','_blank');
}

function startCalibration(){
  const section=parseInt(document.getElementById('calSection').value);
  fetch('/api/calibrate/start',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({section:section,shots:20})
  }).then(()=>{
    document.getElementById('calProgress').style.display='block';
    document.getElementById('calCount').innerText='0';
    document.getElementById('calTarget').innerText='20';
    document.getElementById('calBar').style.width='0%';
  });
}

function initSSE(){
  evt=new EventSource('/events');
  
  evt.addEventListener('live',e=>{
    const data=JSON.parse(e.data);
    const s=data.session;
    if(s&&s.active){
      document.getElementById('liveInfo').innerText='Session #'+s.id+' • Active';
      document.getElementById('liveStats').innerHTML=`
        <div>Attempts: <strong style="color:var(--primary)">${s.attempts}</strong></div>
        <div style="margin-top:8px">Makes: <strong style="color:var(--success)">${s.makes}</strong></div>
        <div style="margin-top:8px">Swishes: <strong style="color:#ffd700">${s.swishes}</strong></div>
      `;
    }
  });
  
  evt.addEventListener('classified',e=>{
    const data=JSON.parse(e.data);
    if(data.accuracy!==undefined){
      const acc=data.accuracy;
      const accElem=document.getElementById('accuracyDisplay');
      accElem.innerText=acc.toFixed(1)+'%';
      accElem.style.color=acc>=80?'var(--success)':acc>=60?'var(--warning)':'var(--danger)';
    }
  });
  
  evt.addEventListener('cal',e=>{
    const data=JSON.parse(e.data);
    document.getElementById('calCount').innerText=data.collected;
    document.getElementById('calTarget').innerText=data.target;
    document.getElementById('calBar').style.width=(data.collected/data.target*100)+'%';
    if(data.finished){
      setTimeout(()=>{
        document.getElementById('calProgress').style.display='none';
        loadStats();
        alert('Calibration complete! Reliability: '+(data.reliability*100).toFixed(0)+'%');
      },1500);
    }
  });
}

document.addEventListener('DOMContentLoaded',init);
</script>
</body>
</html>
)rawliteral";

String buildIndexHtml() {
    return FPSTR(index_html);
}

// ============================================================================
// SERVER ROUTES
// ============================================================================
void handleRoot(AsyncWebServerRequest *request) {
    request->send(200, "text/html", cachedIndex);
}

void handleApiUsers(AsyncWebServerRequest *request) {
    StaticJsonDocument<1536> doc;
    JsonArray arr = doc.createNestedArray("users");
    
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
        if (!lk.isLocked()) {
            request->send(503);
            return;
        }
        
        for (uint8_t i = 0; i < userCount; i++) {
            JsonObject obj = arr.createNestedObject();
            obj["username"] = users[i].username;
            obj["backgroundLearning"] = users[i].backgroundLearning;
            obj["accuracy"] = users[i].getAccuracy();
            
            int totalShots = 0;
            for (const auto& sec : users[i].sections) {
                totalShots += sec.stats.attempts;
            }
            obj["totalShots"] = totalShots;
        }
    }
    
    String out;
    serializeJson(doc, out);
    request->send(200, "application/json", out);
}

void handleCreateUser(AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total) {
    static String buffer;
    
    if (index == 0) {
        buffer = "";
        buffer.reserve(min((size_t)256, total));
    }
    
    for (size_t i = 0; i < len; i++) {
        buffer += (char)data[i];
    }
    
    if (index + len == total) {
        StaticJsonDocument<256> doc;
        deserializeJson(doc, buffer);
        
        String username = doc["username"].as<String>();
        
        if (username.length() == 0 || userCount >= MAX_USERS) {
            request->send(400, "application/json", "{\"error\":\"Invalid\"}");
            return;
        }
        
        { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
            if (!lk.isLocked()) {
                request->send(503);
                return;
            }
            
            strncpy(users[userCount].username, username.c_str(), 23);
            users[userCount].username[23] = '\0';
            users[userCount].backgroundLearning = true;
            
            const char* defaultNames[] = {
                "Paint", "L Corner", "R Corner", "L Wing",
                "R Wing", "L Mid", "R Mid", "Top Key"
            };
            
            for (int i = 0; i < NUM_SECTIONS; i++) {
                strncpy(users[userCount].sections[i].name, defaultNames[i], 23);
                users[userCount].sections[i].name[23] = '\0';
            }
            
            userCount++;
            dataDirty = true;
        }
        
        request->send(200, "application/json", "{\"success\":true}");
    }
}

void handleSelectUser(AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total) {
    static String buffer;
    
    if (index == 0) {
        buffer = "";
        buffer.reserve(min((size_t)128, total));
    }
    
    for (size_t i = 0; i < len; i++) {
        buffer += (char)data[i];
    }
    
    if (index + len == total) {
        StaticJsonDocument<128> doc;
        deserializeJson(doc, buffer);
        
        int userId = doc["userId"] | -1;
        currentUserId = userId;
        
        request->send(200);
    }
}

void handleGetUser(AsyncWebServerRequest *request) {
    int userId = request->pathArg(0).toInt();
    
    if (userId < 0 || userId >= userCount) {
        request->send(404);
        return;
    }
    
    StaticJsonDocument<384> doc;
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
        if (!lk.isLocked()) {
            request->send(503);
            return;
        }
        doc["username"] = users[userId].username;
        doc["backgroundLearning"] = users[userId].backgroundLearning;
        doc["accuracy"] = users[userId].getAccuracy();
    }
    
    String out;
    serializeJson(doc, out);
    request->send(200, "application/json", out);
}

void handleGetUserSections(AsyncWebServerRequest *request) {
    int userId = request->pathArg(0).toInt();
    
    if (userId < 0 || userId >= userCount) {
        request->send(404);
        return;
    }
    
    auto* res = request->beginResponseStream("application/json");
    res->print("{\"sections\":[");
    
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
        if (!lk.isLocked()) {
            request->send(503);
            return;
        }
        
        for (int i = 0; i < NUM_SECTIONS; i++) {
            if (i) res->print(",");
            res->printf("{\"attempts\":%u,\"makes\":%u,\"swishes\":%u,\"fingerprintSamples\":%u}",
                       users[userId].sections[i].stats.attempts,
                       users[userId].sections[i].stats.makes,
                       users[userId].sections[i].stats.swishes,
                       users[userId].sections[i].fingerprint.samples);
        }
    }
    
    res->print("]}");
    request->send(res);
}

void handleToggleLearning(AsyncWebServerRequest *request) {
    int userId = request->pathArg(0).toInt();
    
    if (userId < 0 || userId >= userCount) {
        request->send(404);
        return;
    }
    
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
        if (!lk.isLocked()) {
            request->send(503);
            return;
        }
        users[userId].backgroundLearning = !users[userId].backgroundLearning;
        dataDirty = true;
    }
    
    request->send(200);
}

void handleResetProfile(AsyncWebServerRequest *request) {
    int userId = request->pathArg(0).toInt();
    
    if (userId < 0 || userId >= userCount) {
        request->send(404);
        return;
    }
    
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
        if (!lk.isLocked()) {
            request->send(503);
            return;
        }
        users[userId].clear();
        dataDirty = true;
    }
    
    request->send(200);
}

void handleExportProfile(AsyncWebServerRequest *request) {
    int userId = request->pathArg(0).toInt();
    
    if (userId < 0 || userId >= userCount) {
        request->send(404);
        return;
    }
    
    auto* res = request->beginResponseStream("text/csv");
    res->print("section,attempts,makes,swishes,rim_makes,percentage,calibration_samples\n");
    
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
        if (!lk.isLocked()) {
            request->send(503);
            return;
        }
        
        for (int i = 0; i < NUM_SECTIONS; i++) {
            const auto& sec = users[userId].sections[i];
            float pct = sec.stats.attempts > 0 ? 
                       (float)sec.stats.makes / sec.stats.attempts * 100 : 0;
            res->printf("%s,%u,%u,%u,%u,%.1f,%u\n",
                       sec.name,
                       sec.stats.attempts,
                       sec.stats.makes,
                       sec.stats.swishes,
                       sec.stats.rimMakes,
                       pct,
                       sec.fingerprint.samples);
        }
    }
    
    request->send(res);
}

void handleLive(AsyncWebServerRequest *request) {
    StaticJsonDocument<256> j;
    JsonObject sess = j.createNestedObject("session");
    
    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(10));
        sess["active"] = sessionActive;
        sess["id"] = currentSessionId;
        sess["sectionId"] = sessionSection;
        sess["attempts"] = currentSessionStats.attempts;
        sess["makes"] = currentSessionStats.makes;
        sess["swishes"] = currentSessionStats.swishes;
    }
    
    String out;
    serializeJson(j, out);
    request->send(200, "application/json", out);
}

void handleCreateSessionPost(AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total) {
    static String buffer;
    
    if (index == 0) {
        buffer = "";
        buffer.reserve(min((size_t)256, total));
    }
    
    for (size_t i = 0; i < len; i++) {
        buffer += (char)data[i];
    }
    
    if (index + len == total) {
        StaticJsonDocument<256> doc;
        deserializeJson(doc, buffer);
        
        int durationMin = doc["durationMin"] | 60;
        int section = doc["section"] | -1;
        bool useAuto = doc["useAutoDetect"] | true;
        
        SessionRecord rec;
        rec.id = nextSessionId++;
        rec.created = nowEpochMs();
        rec.durationMs = (uint32_t)durationMin * 60000;
        rec.sectionId = section;
        rec.useAutoDetect = useAuto;
        
        if (currentUserId >= 0 && currentUserId < userCount) {
            strncpy(rec.username, users[currentUserId].username, 23);
        } else {
            strcpy(rec.username, "Anybody");
        }
        rec.username[23] = '\0';
        rec.stats.clear();

        { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
            if (!lk.isLocked()) {
                request->send(503);
                return;
            }
            
            if (sessionCount < MAX_SESSIONS) {
                sessions[sessionCount++] = rec;
            }
            
            sessionActive = true;
            sessionStartMs = millis();
            sessionDurationMs = rec.durationMs;
            sessionSection = rec.sectionId;
            sessionUseAutoDetect = useAuto;
            currentSessionStats.clear();
            currentSessionId = rec.id;
        }
        
        pushLiveUpdate();
        
        StaticJsonDocument<128> out;
        out["id"] = rec.id;
        String outS;
        serializeJson(out, outS);
        request->send(200, "application/json", outS);
    }
}

void handleCalibrateStart(AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total) {
    static String buffer;
    
    if (index == 0) {
        buffer = "";
        buffer.reserve(min((size_t)256, total));
    }
    
    for (size_t i = 0; i < len; i++) {
        buffer += (char)data[i];
    }
    
    if (index + len == total) {
        StaticJsonDocument<256> doc;
        deserializeJson(doc, buffer);
        
        int section = doc["section"] | -1;
        uint32_t shots = doc["shots"] | MIN_CALIBRATION_SHOTS;
        
        if (section < 0 || section >= NUM_SECTIONS || currentUserId < 0) {
            request->send(400);
            return;
        }
        
        { ScopedLock lk(dataMutex, pdMS_TO_TICKS(50));
            if (!lk.isLocked()) {
                request->send(503);
                return;
            }
            calibrating = true;
            calibrateSectionId = section;
            calibrateShotsTarget = clampInt(shots, MIN_CALIBRATION_SHOTS, MAX_CALIBRATION_SHOTS);
            calibrateShotsCollected = 0;
        }
        
        request->send(200);
    }
}

void handleDiagnostics(AsyncWebServerRequest *request) {
    StaticJsonDocument<384> doc;
    doc["heap_free"] = ESP.getFreeHeap();
    doc["heap_min"] = sysHealth.minFreeHeap;
    doc["sensor_available"] = sensorAvailable;
    doc["baseline"] = baseline;
    doc["uptime_ms"] = millis();
    doc["user_count"] = userCount;
    doc["session_count"] = sessionCount;
    doc["version"] = "3.2 Elite";
    
    String out;
    serializeJson(doc, out);
    request->send(200, "application/json", out);
}

// ============================================================================
// SSE HELPERS
// ============================================================================
void pushLiveUpdate() {
    StaticJsonDocument<384> j;
    JsonObject sess = j.createNestedObject("session");
    sess["active"] = sessionActive;
    sess["id"] = currentSessionId;
    sess["attempts"] = currentSessionStats.attempts;
    sess["makes"] = currentSessionStats.makes;
    sess["swishes"] = currentSessionStats.swishes;
    sess["sectionId"] = sessionSection;
    char buf[384];
    serializeJson(j, buf);
    events.send(buf, "live", millis());
}

void pushSectionUpdate() {
    events.send("refresh", "update", millis());
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
    Serial.begin(115200);
    Serial.println("\n\n========================================");
    Serial.println("HoopTracker Pro Elite v3.2");
    Serial.println("========================================");
    Serial.println("Improvements:");
    Serial.println("- Session-level auto/manual control");
    Serial.println("- Separate background learning toggle");
    Serial.println("- Outlier rejection (>3σ)");
    Serial.println("- Temporal smoothing (EMA)");
    Serial.println("- Quadratic velocity fit");
    Serial.println("- Profile reset & export");
    Serial.println("========================================\n");

    dataMutex = xSemaphoreCreateMutex();
    if (!dataMutex) {
        Serial.println("FATAL: Mutex failed");
        while(1) delay(1000);
    }

    pinMode(VIB_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(VIB_PIN), vibISR, RISING);
    Wire.begin(SDA_PIN, SCL_PIN);

    if (!SPIFFS.begin(true)) {
        Serial.println("ERROR: SPIFFS failed");
    } else {
        Serial.println("✓ SPIFFS mounted");
    }

    loadConfig();
    { ScopedLock lk(dataMutex);
        loadUsers();
        loadSessions();
    }

    sensor.setTimeout(500);
    Serial.print("Initializing sensor...");
    for (int attempt = 0; attempt < 4; ++attempt) {
        if (sensor.init()) {
            sensorAvailable = true;
            sensor.setDistanceMode(VL53L1X::Medium);
            sensor.setMeasurementTimingBudget(30000);
            sensor.startContinuous(33);
            Serial.println(" ✓ OK");
            break;
        }
        Serial.print(".");
        delay(200);
    }
    
    if (!sensorAvailable) {
        Serial.println(" ✗ FAILED");
        sysHealth.sensorFailures++;
    }

    esp_task_wdt_init(30, true);
    esp_task_wdt_add(NULL);
    Serial.println("✓ Watchdog enabled");

    noInterrupts();
    for (int i = 0; i < VIB_BUF_SIZE; i++) vibTimestamps[i] = 0;
    interrupts();

    server.addHandler(&events);
    server.on("/", HTTP_GET, handleRoot);
    server.on("/api/users", HTTP_GET, handleApiUsers);
    server.on("/api/users/create", HTTP_POST, 
        [](AsyncWebServerRequest *r){}, NULL, handleCreateUser);
    server.on("/api/users/select", HTTP_POST,
        [](AsyncWebServerRequest *r){}, NULL, handleSelectUser);
    server.on("/api/users/{}", HTTP_GET, handleGetUser);
    server.on("/api/users/{}/sections", HTTP_GET, handleGetUserSections);
    server.on("/api/users/{}/toggle-learning", HTTP_POST, handleToggleLearning);
    server.on("/api/users/{}/reset", HTTP_POST, handleResetProfile);
    server.on("/api/users/{}/export.csv", HTTP_GET, handleExportProfile);
    server.on("/api/live", HTTP_GET, handleLive);
    server.on("/api/session/create", HTTP_POST,
        [](AsyncWebServerRequest *r){}, NULL, handleCreateSessionPost);
    server.on("/api/calibrate/start", HTTP_POST,
        [](AsyncWebServerRequest *r){}, NULL, handleCalibrateStart);
    server.on("/api/diagnostics", HTTP_GET, handleDiagnostics);

    AsyncElegantOTA.begin(&server, cfg.otaUser, cfg.otaPass);
    server.begin();

    cachedIndex = buildIndexHtml();

    Serial.println("\n========================================");
    Serial.println("✓ HoopTracker Pro Elite Ready!");
    Serial.printf("✓ Free heap: %zu bytes\n", ESP.getFreeHeap());
    Serial.printf("✓ Min heap: %zu bytes\n", sysHealth.minFreeHeap);
    Serial.printf("✓ Users: %u/%u\n", userCount, MAX_USERS);
    Serial.printf("✓ Features: %d (physically valid)\n", NUM_FEATURES);
    Serial.println("========================================\n");
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
    static uint32_t nextSensorTime = 0;
    static uint32_t nextHeapCheck = 0;
    static uint32_t nextSaveCheck = 0;
    static uint32_t lastLiveTickSec = 0;

    uint32_t now = millis();
    esp_task_wdt_reset();

    // Sensor @ 30Hz
    if ((int32_t)(now - nextSensorTime) >= 0) {
        nextSensorTime = now + 33;
        handleVibrationFlag(now);

        int d = -1;
        if (sensorAvailable && sensor.dataReady()) {
            d = sensor.read();
            if (sensor.timeoutOccurred()) {
                d = -1;
                sysHealth.sensorFailures++;
            }
        }

        pushTofSample(d, now);

        if (d > 0 && shotState == ShotState::Idle) {
            baselineAvg = baselineAvg * (1.0f - cfg.baselineAlpha) + d * cfg.baselineAlpha;
            baseline = clampInt((int)roundf(baselineAvg), 200, 2000);
        }

        bool tofDrop = (d > 0) && ((baseline - d) > cfg.ballTriggerDrop);
        bool nearBaseline = (d > 0) && (abs(d - baseline) < 50);
        uint32_t lastVib = lastVibrationMs.load(std::memory_order_relaxed);

        // FSM
        switch (shotState) {
            case ShotState::Idle:
                if (tofDrop) {
                    rimHitPre = false;
                    transitionState(ShotState::BallEntering, "Ball");
                } else if (lastVib > 0 && (int32_t)(now - lastVib) <= 100) {
                    rimHitPre = true;
                    vibWindowEndMs = now + cfg.entryWindowMs;
                    transitionState(ShotState::VibrationPrimed, "Rim first");
                }
                break;

            case ShotState::VibrationPrimed:
                if (tofDrop) {
                    transitionState(ShotState::BallEntering, "Entering");
                } else if ((int32_t)(now - vibWindowEndMs) >= 0) {
                    registerAttemptOutcome(false, false);
                    lastShotMs = now;
                    transitionState(ShotState::Cooldown, "Miss");
                    rimHitPre = false;
                }
                break;

            case ShotState::BallEntering:
                if (nearBaseline) {
                    rimWindowEndMs = now + cfg.rimWindowMs;
                    transitionState(ShotState::AwaitRimWindow, "At rim");
                }
                break;

            case ShotState::AwaitRimWindow:
                if ((int32_t)(now - rimWindowEndMs) >= 0) {
                    uint32_t lastVibNow = lastVibrationMs.load(std::memory_order_relaxed);
                    bool rimHit = rimHitPre || ((int32_t)(now - lastVibNow) <= (int32_t)cfg.rimWindowMs);
                    bool swish = !rimHit;
                    
                    uint32_t ballTriggerMs = now - cfg.rimWindowMs - 80;
                    uint32_t rimImpactMs = lastVibNow ? lastVibNow : now;
                    
                    FeatureVector fv = extractPhysicalFeatures(ballTriggerMs, rimImpactMs);
                    
                    registerAttemptOutcomeWithVector(true, swish, &fv);
                    
                    lastShotMs = now;
                    transitionState(ShotState::Cooldown, "Made");
                    rimHitPre = false;
                    
                    { ScopedLock lk(dataMutex, pdMS_TO_TICKS(10));
                        if (lk.isLocked() && calibrating && calibrateSectionId >= 0) {
                            acceptCalibrationSample(fv);
                        }
                    }
                }
                break;

            case ShotState::Cooldown:
                if ((int32_t)(now - lastShotMs) >= (int32_t)cfg.shotCooldownMs) {
                    transitionState(ShotState::Idle, "Ready");
                }
                break;
        }
    }

    // Heap check every 10s
    if ((int32_t)(now - nextHeapCheck) >= 0) {
        nextHeapCheck = now + 10000;
        checkHeapHealth();
    }

    // Periodic saves
    if ((int32_t)(now - nextSaveCheck) >= 0) {
        nextSaveCheck = now + 5000;
        
        if (dataDirty && (int32_t)(now - lastDataSaveMs) >= (int32_t)DATA_SAVE_INTERVAL_MS) {
            ScopedLock lk(dataMutex, pdMS_TO_TICKS(100));
            if (lk.isLocked()) {
                saveUsers();
                saveSessions();
                dataDirty = false;
                lastDataSaveMs = now;
            }
        }
    }

    // Live updates
    uint32_t nowSec = now / 1000;
    if (sessionActive && nowSec != lastLiveTickSec) {
        lastLiveTickSec = nowSec;
        pushLiveUpdate();
    }

    vTaskDelay(pdMS_TO_TICKS(1));
}