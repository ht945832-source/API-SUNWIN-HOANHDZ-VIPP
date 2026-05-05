const WebSocket = require('ws');
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const HttpsProxyAgent = require('https-proxy-agent');

const app = express();
app.use(cors());
const PORT = process.env.PORT || 3001;

// ==================== PROXY ROTATION (GIẢ IP) ====================
// Danh sách proxy xoay vòng để tránh bị chặn
const proxyList = [
    'http://45.76.179.21:8080',
    'http://103.28.36.235:8080',
    'http://103.143.176.10:8080',
    'http://103.28.36.235:8080',
    'http://45.76.179.21:8080',
    // Thêm proxy free hoặc trả phí ở đây
];

let currentProxyIndex = 0;

function getNextProxy() {
    currentProxyIndex = (currentProxyIndex + 1) % proxyList.length;
    return proxyList[currentProxyIndex];
}

function createProxyAgent() {
    if (proxyList.length === 0) return null;
    const proxyUrl = getNextProxy();
    return new HttpsProxyAgent(proxyUrl);
}

// ==================== FILE STORAGE ====================
const HISTORY_FILE = './history.json';
const PATTERNS_FILE = './patterns.json';
const MODEL_WEIGHTS_FILE = './model_weights.json';
const PREDICTION_MODEL_FILE = './prediction_model.json';

// Load history if exists
let resultHistory = [];
if (fs.existsSync(HISTORY_FILE)) {
    try {
        resultHistory = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'));
        console.log(`[📂] Đã tải ${resultHistory.length} phiên từ history.json`);
    } catch (e) {
        console.error('[❌] Lỗi đọc history.json:', e.message);
    }
}

// Load prediction model
let predictionModel = {
    // Thống kê tần suất các pattern
    patternStats: {},
    // Tỉ lệ thắng của từng loại dự đoán
    predictionAccuracy: {
        'Tài': { total: 0, correct: 0 },
        'Xỉu': { total: 0, correct: 0 }
    },
    // Bộ nhớ các chuỗi kết quả
    sequenceMemory: {},
    // Chiến thuật hiện tại
    currentStrategy: 'balanced',
    // Last 10 predictions for analysis
    recentPredictions: []
};

if (fs.existsSync(PREDICTION_MODEL_FILE)) {
    try {
        const saved = JSON.parse(fs.readFileSync(PREDICTION_MODEL_FILE, 'utf8'));
        predictionModel = { ...predictionModel, ...saved };
        console.log('[📂] Đã tải prediction_model.json');
    } catch (e) {
        console.error('[❌] Lỗi đọc prediction_model.json:', e.message);
    }
}

// Load model weights if exists
let modelWeights = {
    'model1': 1.0, 'model2': 1.0, 'model3': 1.0, 'model4': 1.0,
    'model5': 1.0, 'model6': 1.0, 'model7': 1.0, 'model8': 1.0,
    'model9': 1.0, 'model10': 1.0, 'model11': 1.0, 'model12': 1.0,
    'model13': 1.0, 'model14': 1.0, 'model15': 1.0, 'model16': 1.0,
    'model17': 1.0, 'model18': 1.0, 'model19': 1.0, 'model20': 1.0,
    'model21': 1.0
};

if (fs.existsSync(MODEL_WEIGHTS_FILE)) {
    try {
        const savedWeights = JSON.parse(fs.readFileSync(MODEL_WEIGHTS_FILE, 'utf8'));
        modelWeights = savedWeights.modelWeights || modelWeights;
        console.log('[📂] Đã tải model_weights.json');
    } catch (e) {
        console.error('[❌] Lỗi đọc model_weights.json:', e.message);
    }
}

// Save history
function saveHistory(entry) {
    resultHistory.push(entry);
    if (resultHistory.length > 1000) resultHistory.shift();
    fs.writeFileSync(HISTORY_FILE, JSON.stringify(resultHistory, null, 2));
}

// Save model weights
function saveModelWeights() {
    const weights = { modelWeights };
    fs.writeFileSync(MODEL_WEIGHTS_FILE, JSON.stringify(weights, null, 2));
}

// Save prediction model
function savePredictionModel() {
    fs.writeFileSync(PREDICTION_MODEL_FILE, JSON.stringify(predictionModel, null, 2));
}

// ==================== GLOBAL VARIABLES ====================
let currentSessionId = null;
let lastResult = null;
let lastPrediction = null;
let stats = {
    total: 0,
    correct: 0,
    wrong: 0,
    consecutiveLosses: 0,
    modelPerformance: {},
    strategyPerformance: {
        'balanced': { total: 0, correct: 0 },
        'aggressive': { total: 0, correct: 0 },
        'conservative': { total: 0, correct: 0 }
    }
};

let apiResponseData = {
    "Phien": null,
    "Xuc_xac_1": null,
    "Xuc_xac_2": null,
    "Xuc_xac_3": null,
    "Tong": null,
    "Ket_qua": "",
    "Phien_hien_tai": null,
    "Du_doan": "",
    "Loai_cau": "",
    "Mau_cau_phat_hien": "",
    "Do_tin_cay": "0%",
    "Trang_thai": "",
    "Ket_qua_du_doan": "",
    "Thong_ke": {
        "tong": 0,
        "dung": 0,
        "sai": 0,
        "ti_le": "0%"
    },
    "id": "@tranhoang2286"
};

// ==================== SMART PREDICTION ENGINE ====================
class SmartPredictionEngine {
    constructor() {
        this.learningRate = 0.05;
        this.minConfidence = 0.48;
        this.maxConfidence = 0.95;
    }

    // Phân tích pattern phức hợp (không dùng kết quả phiên trước đơn thuần)
    analyzeComplexPatterns(history) {
        if (history.length < 8) {
            return { prediction: null, confidence: 0, reason: 'Đang thu thập dữ liệu...' };
        }

        const results = history.map(h => h.Ket_qua);
        const scores = history.map(h => h.Tong);
        
        // ========== 1. PHÂN TÍCH CHU KỲ HIDDEN ==========
        const cycleAnalysis = this.detectHiddenCycles(results);
        
        // ========== 2. PHÂN TÍCH TƯƠNG QUAN ĐIỂM SỐ ==========
        const scoreCorrelation = this.analyzeScoreCorrelation(scores, results);
        
        // ========== 3. PHÂN TÍCH NHỊP ĐIỆU ==========
        const rhythmAnalysis = this.analyzeRhythm(results);
        
        // ========== 4. PHÂN TÍCH MA TRẬN CHUYỂN TIẾP ==========
        const transitionMatrix = this.buildTransitionMatrix(results);
        
        // ========== 5. PHÂN TÍCH XÁC SUẤT THỐNG KÊ ==========
        const statisticalProb = this.statisticalProbability(results);
        
        // ========== 6. PHÂN TÍCH MẪU CẦU ĐẸP ==========
        const beautifulPattern = this.detectBeautifulPattern(results);
        
        // ========== 7. PHÂN TÍCH CHUỖI FIBONACCI ==========
        const fibonacciPattern = this.detectFibonacciPattern(results);
        
        // ========== 8. PHÂN TÍCH ĐỐI XỨNG ==========
        const symmetryPattern = this.detectSymmetry(results);
        
        // ========== 9. PHÂN TÍCH XU HƯỚNG ĐA CHIỀU ==========
        const multiTrend = this.multiDimensionalTrend(results, scores);
        
        // Tổng hợp tất cả các phân tích
        let taiScore = 0;
        let xiuScore = 0;
        let totalWeight = 0;
        let reasons = [];

        // Cycle analysis
        if (cycleAnalysis.prediction) {
            const weight = 0.15;
            if (cycleAnalysis.prediction === 'Tài') taiScore += weight * cycleAnalysis.confidence;
            else xiuScore += weight * cycleAnalysis.confidence;
            totalWeight += weight;
            reasons.push(`Chu kỳ ẩn: ${cycleAnalysis.reason}`);
        }

        // Score correlation
        if (scoreCorrelation.prediction) {
            const weight = 0.12;
            if (scoreCorrelation.prediction === 'Tài') taiScore += weight * scoreCorrelation.confidence;
            else xiuScore += weight * scoreCorrelation.confidence;
            totalWeight += weight;
            reasons.push(`Tương quan điểm: ${scoreCorrelation.reason}`);
        }

        // Rhythm analysis
        if (rhythmAnalysis.prediction) {
            const weight = 0.1;
            if (rhythmAnalysis.prediction === 'Tài') taiScore += weight * rhythmAnalysis.confidence;
            else xiuScore += weight * rhythmAnalysis.confidence;
            totalWeight += weight;
            reasons.push(`Nhịp điệu: ${rhythmAnalysis.reason}`);
        }

        // Transition matrix
        if (transitionMatrix.prediction) {
            const weight = 0.13;
            if (transitionMatrix.prediction === 'Tài') taiScore += weight * transitionMatrix.confidence;
            else xiuScore += weight * transitionMatrix.confidence;
            totalWeight += weight;
            reasons.push(`Ma trận chuyển tiếp: ${transitionMatrix.reason}`);
        }

        // Statistical probability
        if (statisticalProb.prediction) {
            const weight = 0.1;
            if (statisticalProb.prediction === 'Tài') taiScore += weight * statisticalProb.confidence;
            else xiuScore += weight * statisticalProb.confidence;
            totalWeight += weight;
            reasons.push(`Xác suất thống kê: ${statisticalProb.reason}`);
        }

        // Beautiful pattern
        if (beautifulPattern.prediction) {
            const weight = 0.12;
            if (beautifulPattern.prediction === 'Tài') taiScore += weight * beautifulPattern.confidence;
            else xiuScore += weight * beautifulPattern.confidence;
            totalWeight += weight;
            reasons.push(`Mẫu cầu đẹp: ${beautifulPattern.reason}`);
        }

        // Fibonacci pattern
        if (fibonacciPattern.prediction) {
            const weight = 0.1;
            if (fibonacciPattern.prediction === 'Tài') taiScore += weight * fibonacciPattern.confidence;
            else xiuScore += weight * fibonacciPattern.confidence;
            totalWeight += weight;
            reasons.push(`Fibonacci: ${fibonacciPattern.reason}`);
        }

        // Symmetry pattern
        if (symmetryPattern.prediction) {
            const weight = 0.08;
            if (symmetryPattern.prediction === 'Tài') taiScore += weight * symmetryPattern.confidence;
            else xiuScore += weight * symmetryPattern.confidence;
            totalWeight += weight;
            reasons.push(`Đối xứng: ${symmetryPattern.reason}`);
        }

        // Multi-dimensional trend
        if (multiTrend.prediction) {
            const weight = 0.1;
            if (multiTrend.prediction === 'Tài') taiScore += weight * multiTrend.confidence;
            else xiuScore += weight * multiTrend.confidence;
            totalWeight += weight;
            reasons.push(`Xu hướng đa chiều: ${multiTrend.reason}`);
        }

        // Quyết định cuối cùng
        if (totalWeight === 0) {
            return { prediction: null, confidence: 0, reason: 'Chưa đủ dữ liệu phân tích' };
        }

        const taiRatio = taiScore / totalWeight;
        const xiuRatio = xiuScore / totalWeight;
        
        let finalPrediction, finalConfidence;
        if (taiRatio > xiuRatio) {
            finalPrediction = 'Tài';
            finalConfidence = Math.min(taiRatio, this.maxConfidence);
        } else {
            finalPrediction = 'Xỉu';
            finalConfidence = Math.min(xiuRatio, this.maxConfidence);
        }

        // Đảm bảo confidence không dưới 48%
        finalConfidence = Math.max(finalConfidence, this.minConfidence);

        return {
            prediction: finalPrediction,
            confidence: finalConfidence,
            reason: reasons.join(' | '),
            details: {
                taiScore: taiScore.toFixed(3),
                xiuScore: xiuScore.toFixed(3),
                totalWeight: totalWeight.toFixed(3)
            }
        };
    }

    // Phát hiện chu kỳ ẩn (không phải pattern đơn giản)
    detectHiddenCycles(results) {
        if (results.length < 12) return { prediction: null, confidence: 0 };
        
        // Tìm chu kỳ từ 2-7 phiên
        for (let cycleLen = 2; cycleLen <= 7; cycleLen++) {
            if (results.length < cycleLen * 2) continue;
            
            let matchCount = 0;
            const lastCycle = results.slice(-cycleLen);
            
            for (let i = 0; i <= results.length - cycleLen * 2; i += cycleLen) {
                const compareCycle = results.slice(i, i + cycleLen);
                if (this.arraysEqual(lastCycle, compareCycle)) {
                    matchCount++;
                }
            }
            
            if (matchCount >= 2) {
                const nextPrediction = lastCycle[0];
                const confidence = 0.5 + (matchCount * 0.05);
                return {
                    prediction: nextPrediction,
                    confidence: Math.min(confidence, 0.75),
                    reason: `Chu kỳ ${cycleLen} phiên lặp lại ${matchCount} lần`
                };
            }
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Phân tích tương quan với điểm số
    analyzeScoreCorrelation(scores, results) {
        if (scores.length < 8) return { prediction: null, confidence: 0 };
        
        const recentScores = scores.slice(-5);
        const avgScore = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;
        const lastScore = scores[scores.length - 1];
        
        // Phân tích xu hướng điểm
        let trend = 0;
        for (let i = 1; i < recentScores.length; i++) {
            trend += recentScores[i] - recentScores[i-1];
        }
        
        // Dự đoán dựa trên xu hướng điểm
        if (trend > 3) {
            // Điểm tăng mạnh -> khả năng Xỉu (điểm sẽ giảm)
            return {
                prediction: 'Xỉu',
                confidence: 0.55 + Math.min(trend * 0.02, 0.2),
                reason: `Điểm tăng mạnh (trend ${trend.toFixed(1)}), dự đoán Xỉu`
            };
        } else if (trend < -3) {
            // Điểm giảm mạnh -> khả năng Tài (điểm sẽ tăng)
            return {
                prediction: 'Tài',
                confidence: 0.55 + Math.min(Math.abs(trend) * 0.02, 0.2),
                reason: `Điểm giảm mạnh (trend ${trend.toFixed(1)}), dự đoán Tài`
            };
        }
        
        // Phân tích điểm bất thường
        const scoreHistory = scores.slice(-20);
        const mean = scoreHistory.reduce((a, b) => a + b, 0) / scoreHistory.length;
        const stdDev = Math.sqrt(scoreHistory.map(s => Math.pow(s - mean, 2)).reduce((a, b) => a + b, 0) / scoreHistory.length);
        
        if (lastScore > mean + stdDev * 1.5) {
            return {
                prediction: 'Xỉu',
                confidence: 0.6,
                reason: `Điểm ${lastScore} cao bất thường (TB ${mean.toFixed(1)})`
            };
        } else if (lastScore < mean - stdDev * 1.5) {
            return {
                prediction: 'Tài',
                confidence: 0.6,
                reason: `Điểm ${lastScore} thấp bất thường (TB ${mean.toFixed(1)})`
            };
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Phân tích nhịp điệu (thay đổi luân phiên)
    analyzeRhythm(results) {
        if (results.length < 6) return { prediction: null, confidence: 0 };
        
        const recent = results.slice(-8);
        let rhythmStrength = 0;
        let lastChange = 0;
        
        for (let i = 1; i < recent.length; i++) {
            if (recent[i] !== recent[i-1]) {
                rhythmStrength++;
                lastChange = i;
            }
        }
        
        const rhythmRatio = rhythmStrength / (recent.length - 1);
        
        if (rhythmRatio > 0.7) {
            // Nhịp điệu cao -> đang trong cầu 1-1
            const last = results[results.length - 1];
            const prediction = last === 'Tài' ? 'Xỉu' : 'Tài';
            return {
                prediction: prediction,
                confidence: 0.6 + rhythmRatio * 0.2,
                reason: `Nhịp điệu cao (${(rhythmRatio*100).toFixed(0)}%)`
            };
        } else if (rhythmRatio < 0.3) {
            // Nhịp điệu thấp -> đang bệt
            const last = results[results.length - 1];
            return {
                prediction: last,
                confidence: 0.55 + (1 - rhythmRatio) * 0.2,
                reason: `Nhịp điệu thấp (${(rhythmRatio*100).toFixed(0)}%)`
            };
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Xây dựng ma trận chuyển tiếp
    buildTransitionMatrix(results) {
        if (results.length < 10) return { prediction: null, confidence: 0 };
        
        // Ma trận chuyển tiếp 2 bậc
        const transitions = {};
        const outcomes = ['Tài', 'Xỉu'];
        
        outcomes.forEach(o1 => {
            outcomes.forEach(o2 => {
                transitions[`${o1}→${o2}`] = { count: 0, next: { 'Tài': 0, 'Xỉu': 0 } };
            });
        });
        
        // Đếm các chuyển tiếp
        for (let i = 2; i < results.length; i++) {
            const key = `${results[i-2]}→${results[i-1]}`;
            if (transitions[key]) {
                transitions[key].count++;
                transitions[key].next[results[i]]++;
            }
        }
        
        // Dự đoán dựa trên 2 kết quả gần nhất
        if (results.length >= 2) {
            const lastKey = `${results[results.length-2]}→${results[results.length-1]}`;
            const transition = transitions[lastKey];
            
            if (transition && transition.count >= 2) {
                const taiProb = transition.next['Tài'] / transition.count;
                const xiuProb = transition.next['Xỉu'] / transition.count;
                
                if (taiProb > 0.6) {
                    return {
                        prediction: 'Tài',
                        confidence: 0.55 + taiProb * 0.2,
                        reason: `Pattern ${lastKey} → Tài (${(taiProb*100).toFixed(0)}%)`
                    };
                } else if (xiuProb > 0.6) {
                    return {
                        prediction: 'Xỉu',
                        confidence: 0.55 + xiuProb * 0.2,
                        reason: `Pattern ${lastKey} → Xỉu (${(xiuProb*100).toFixed(0)}%)`
                    };
                }
            }
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Xác suất thống kê cơ bản
    statisticalProbability(results) {
        if (results.length < 20) return { prediction: null, confidence: 0 };
        
        const last20 = results.slice(-20);
        const taiCount = last20.filter(r => r === 'Tài').length;
        const xiuCount = 20 - taiCount;
        
        // Tính độ lệch so với cân bằng
        const deviation = Math.abs(taiCount - xiuCount) / 20;
        
        if (deviation > 0.3) {
            if (taiCount > xiuCount) {
                return {
                    prediction: 'Xỉu',
                    confidence: 0.5 + deviation * 0.2,
                    reason: `Tài vượt ${taiCount}/20, dự đoán Xỉu cân bằng`
                };
            } else {
                return {
                    prediction: 'Tài',
                    confidence: 0.5 + deviation * 0.2,
                    reason: `Xỉu vượt ${xiuCount}/20, dự đoán Tài cân bằng`
                };
            }
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Phát hiện mẫu cầu đẹp (pattern đặc biệt)
    detectBeautifulPattern(results) {
        if (results.length < 8) return { prediction: null, confidence: 0 };
        
        const last8 = results.slice(-8);
        
        // Pattern: 3-2-3 (T T T X X T T T)
        if (last8[0] === last8[1] && last8[1] === last8[2] &&
            last8[3] === last8[4] && last8[5] === last8[6] && last8[6] === last8[7]) {
            return {
                prediction: last8[4],
                confidence: 0.75,
                reason: 'Phát hiện cầu 3-2-3'
            };
        }
        
        // Pattern: 2-4-2 (T T X X X X T T)
        if (last8[0] === last8[1] && last8[2] === last8[3] && last8[3] === last8[4] && 
            last8[4] === last8[5] && last8[6] === last8[7]) {
            const middleValue = last8[2];
            const prediction = middleValue === 'Tài' ? 'Xỉu' : 'Tài';
            return {
                prediction: prediction,
                confidence: 0.7,
                reason: 'Phát hiện cầu 2-4-2'
            };
        }
        
        // Pattern: 1-3-1-3
        if (results.length >= 10) {
            const last10 = results.slice(-10);
            let patternValid = true;
            for (let i = 0; i < 9; i += 2) {
                if (last10[i] !== last10[i+2]) {
                    patternValid = false;
                    break;
                }
            }
            if (patternValid && last10[0] !== last10[1]) {
                return {
                    prediction: last10[8],
                    confidence: 0.7,
                    reason: 'Phát hiện cầu 1-3-1-3'
                };
            }
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Phát hiện pattern Fibonacci
    detectFibonacciPattern(results) {
        if (results.length < 13) return { prediction: null, confidence: 0 };
        
        // Fibonacci distances: 1, 2, 3, 5, 8
        const fibDistances = [1, 2, 3, 5, 8];
        
        for (const dist of fibDistances) {
            if (results.length > dist * 2) {
                const last = results[results.length - 1];
                const atDist = results[results.length - 1 - dist];
                
                if (last === atDist) {
                    const prevAtDist = results[results.length - 1 - dist * 2];
                    if (prevAtDist && prevAtDist === last) {
                        return {
                            prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.65,
                            reason: `Pattern Fibonacci khoảng cách ${dist}`
                        };
                    }
                }
            }
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Phát hiện cầu đối xứng
    detectSymmetry(results) {
        if (results.length < 7) return { prediction: null, confidence: 0 };
        
        // Kiểm tra đối xứng quanh tâm
        for (let radius = 1; radius <= 3; radius++) {
            let symmetric = true;
            for (let i = 1; i <= radius; i++) {
                if (results[results.length - i] !== results[results.length - radius * 2 + i - 1]) {
                    symmetric = false;
                    break;
                }
            }
            
            if (symmetric && radius >= 2) {
                const centerIndex = results.length - radius;
                if (centerIndex >= 0) {
                    return {
                        prediction: results[centerIndex],
                        confidence: 0.65 + radius * 0.05,
                        reason: `Cầu đối xứng bán kính ${radius}`
                    };
                }
            }
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Phân tích xu hướng đa chiều
    multiDimensionalTrend(results, scores) {
        if (results.length < 10) return { prediction: null, confidence: 0 };
        
        let taiMomentum = 0;
        let xiuMomentum = 0;
        
        // Xu hướng 5 phiên gần nhất
        const last5 = results.slice(-5);
        for (let i = 1; i < last5.length; i++) {
            if (last5[i] === 'Tài') taiMomentum++;
            else xiuMomentum++;
        }
        
        // Xu hướng điểm
        if (scores.length >= 5) {
            const last5Scores = scores.slice(-5);
            const scoreTrend = last5Scores[4] - last5Scores[0];
            if (scoreTrend > 0) taiMomentum += 0.5;
            else if (scoreTrend < 0) xiuMomentum += 0.5;
        }
        
        // Xu hướng dài hạn
        const last20 = results.slice(-20);
        const longTai = last20.filter(r => r === 'Tài').length;
        if (longTai > 12) xiuMomentum += 1;
        else if (longTai < 8) taiMomentum += 1;
        
        if (taiMomentum > xiuMomentum + 1.5) {
            return {
                prediction: 'Tài',
                confidence: 0.58,
                reason: `Xu hướng Tài chiếm ưu thế`
            };
        } else if (xiuMomentum > taiMomentum + 1.5) {
            return {
                prediction: 'Xỉu',
                confidence: 0.58,
                reason: `Xu hướng Xỉu chiếm ưu thế`
            };
        }
        
        return { prediction: null, confidence: 0 };
    }

    // Cập nhật model dựa trên kết quả thực tế
    updateModel(actualResult, predictedResult, confidence) {
        const isCorrect = actualResult === predictedResult;
        
        // Cập nhật tỉ lệ chính xác cho loại dự đoán
        if (predictionModel.predictionAccuracy[predictedResult]) {
            predictionModel.predictionAccuracy[predictedResult].total++;
            if (isCorrect) {
                predictionModel.predictionAccuracy[predictedResult].correct++;
            }
        }
        
        // Cập nhật bộ nhớ sequence
        if (resultHistory.length >= 2) {
            const lastTwo = resultHistory.slice(-2).map(h => h.Ket_qua);
            const key = `${lastTwo[0]}→${lastTwo[1]}`;
            if (!predictionModel.sequenceMemory[key]) {
                predictionModel.sequenceMemory[key] = { 'Tài': 0, 'Xỉu': 0 };
            }
            predictionModel.sequenceMemory[key][actualResult]++;
        }
        
        // Lưu dự đoán gần đây
        predictionModel.recentPredictions.push({
            predicted: predictedResult,
            actual: actualResult,
            correct: isCorrect,
            confidence: confidence,
            timestamp: Date.now()
        });
        
        // Giữ 50 dự đoán gần nhất
        if (predictionModel.recentPredictions.length > 50) {
            predictionModel.recentPredictions.shift();
        }
        
        // Điều chỉnh chiến thuật dựa trên hiệu suất gần đây
        this.adjustStrategy();
        
        savePredictionModel();
    }
    
    // Điều chỉnh chiến thuật
    adjustStrategy() {
        const recent = predictionModel.recentPredictions.slice(-20);
        if (recent.length < 10) return;
        
        const correctCount = recent.filter(p => p.correct).length;
        const accuracy = correctCount / recent.length;
        
        if (accuracy >= 0.6) {
            predictionModel.currentStrategy = 'aggressive';
        } else if (accuracy <= 0.45) {
            predictionModel.currentStrategy = 'conservative';
        } else {
            predictionModel.currentStrategy = 'balanced';
        }
        
        // Cập nhật learning rate dựa trên hiệu suất
        this.learningRate = 0.03 + (1 - accuracy) * 0.04;
        
        console.log(`[📊] Chiến thuật: ${predictionModel.currentStrategy} | Acc gần đây: ${(accuracy*100).toFixed(1)}% | LR: ${this.learningRate.toFixed(3)}`);
    }
    
    // Lấy độ tin cậy điều chỉnh theo chiến thuật
    getAdjustedConfidence(baseConfidence) {
        let adjusted = baseConfidence;
        
        switch (predictionModel.currentStrategy) {
            case 'aggressive':
                adjusted = Math.min(baseConfidence * 1.1, this.maxConfidence);
                break;
            case 'conservative':
                adjusted = Math.max(baseConfidence * 0.9, this.minConfidence);
                break;
            default:
                adjusted = baseConfidence;
        }
        
        return Math.max(adjusted, this.minConfidence);
    }
    
    arraysEqual(a, b) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) return false;
        }
        return true;
    }
}

// Initialize smart engine
const smartEngine = new SmartPredictionEngine();

// ==================== WEBSOCKET ====================
const WEBSOCKET_URL = "wss://websocket.azhkthg1.net/websocket?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhbW91bnQiOjAsInVzZXJuYW1lIjoiU0NfYXBpc3Vud2luMTIzIn0.hgrRbSV6vnBwJMg9ZFtbx3rRu9mX_hZMZ_m5gMNhkw0";

const RECONNECT_DELAY = 2500;
const PING_INTERVAL = 15000;

const initialMessages = [
    [
        1,
        "MiniGame",
        "GM_apivopnha",
        "WangLin",
        {
            "info": "{\"ipAddress\":\"14.249.227.107\",\"wsToken\":\"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJnZW5kZXIiOjAsImNhblZpZXdTdGF0IjpmYWxzZSwiZGlzcGxheU5hbWUiOiI5ODE5YW5zc3MiLCJib3QiOjAsImlzTWVyY2hhbnQiOmZhbHNlLCJ2ZXJpZmllZEJhbmtBY2NvdW50IjpmYWxzZSwicGxheUV2ZW50TG9iYnkiOmZhbHNlLCJjdXN0b21lcklkIjozMjMyODExNTEsImFmZklkIjoic3VuLndpbiIsImJhbm5lZCI6ZmFsc2UsImJyYW5kIjoiZ2VtIiwidGltZXN0YW1wIjoxNzYzMDMyOTI4NzcwLCJsb2NrR2FtZXMiOltdLCJhbW91bnQiOjAsImxvY2tDaGF0IjpmYWxzZSwicGhvbmVWZXJpZmllZCI6ZmFsc2UsImlwQWRkcmVzcyI6IjE0LjI0OS4yMjcuMTA3IiwibXV0ZSI6ZmFsc2UsImF2YXRhciI6Imh0dHBzOi8vaW1hZ2VzLnN3aW5zaG9wLm5ldC9pbWFnZXMvYXZhdGFyL2F2YXRhcl8wNS5wbmciLCJwbGF0Zm9ybUlkIjo0LCJ1c2VySWQiOiI4ODM4NTMzZS1kZTQzLTRiOGQtOTUwMy02MjFmNDA1MDUzNGUiLCJyZWdUaW1lIjoxNzYxNjMyMzAwNTc2LCJwaG9uZSI6IiIsImRlcG9zaXQiOmZhbHNlLCJ1c2VybmFtZSI6IkdNX2FwaXZvcG5oYSJ9.guH6ztJSPXUL1cU8QdMz8O1Sdy_SbxjSM-CDzWPTr-0\",\"locale\":\"vi\",\"userId\":\"8838533e-de43-4b8d-9503-621f4050534e\",\"username\":\"GM_apivopnha\",\"timestamp\":1763032928770,\"refreshToken\":\"e576b43a64e84f789548bfc7c4c8d1e5.7d4244a361e345908af95ee2e8ab2895\"}",
            "signature": "45EF4B318C883862C36E1B189A1DF5465EBB60CB602BA05FAD8FCBFCD6E0DA8CB3CE65333EDD79A2BB4ABFCE326ED5525C7D971D9DEDB5A17A72764287FFE6F62CBC2DF8A04CD8EFF8D0D5AE27046947ADE45E62E644111EFDE96A74FEC635A97861A425FF2B5732D74F41176703CA10CFEED67D0745FF15EAC1065E1C8BCBFA"
        }
    ],
    [6, "MiniGame", "taixiuPlugin", { cmd: 1005 }],
    [6, "MiniGame", "lobbyPlugin", { cmd: 10001 }]
];

let ws = null;
let pingInterval = null;
let reconnectTimeout = null;

function connectWebSocket() {
    if (ws) {
        ws.removeAllListeners();
        ws.close();
    }

    // Sử dụng proxy nếu có
    const options = {};
    const proxyAgent = createProxyAgent();
    if (proxyAgent) {
        options.agent = proxyAgent;
        console.log('[🔄] Sử dụng proxy để giả IP:', proxyList[currentProxyIndex]);
    }

    ws = new WebSocket(WEBSOCKET_URL, options);

    ws.on('open', () => {
        console.log('[✅] WebSocket connected.');
        initialMessages.forEach((msg, i) => {
            setTimeout(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(msg));
                }
            }, i * 600);
        });

        clearInterval(pingInterval);
        pingInterval = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.ping();
            }
        }, PING_INTERVAL);
    });

    ws.on('pong', () => {});

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            if (!Array.isArray(data) || typeof data[1] !== 'object') return;
            const { cmd, sid, d1, d2, d3, gBB } = data[1];

            if (cmd === 1008 && sid) {
                currentSessionId = sid;
            }

            if (cmd === 1003 && gBB) {
                if (!d1 || !d2 || !d3) return;

                const total = d1 + d2 + d3;
                const result = (total > 10) ? "Tài" : "Xỉu";

                // Kiểm tra dự đoán cũ
                let predictionCorrect = false;
                if (lastPrediction && lastPrediction.ket_qua) {
                    predictionCorrect = (lastPrediction.ket_qua === result);
                    
                    // Update stats
                    stats.total++;
                    if (predictionCorrect) {
                        stats.correct++;
                        stats.consecutiveLosses = 0;
                    } else {
                        stats.wrong++;
                        stats.consecutiveLosses++;
                    }
                    
                    // Update strategy performance
                    const strategy = predictionModel.currentStrategy;
                    if (stats.strategyPerformance[strategy]) {
                        stats.strategyPerformance[strategy].total++;
                        if (predictionCorrect) {
                            stats.strategyPerformance[strategy].correct++;
                        }
                    }
                    
                    // Update smart engine
                    smartEngine.updateModel(result, lastPrediction.ket_qua, 
                        parseFloat(lastPrediction.do_tin_cay) / 100);
                }

                // Lưu lịch sử phiên này
                const historyEntry = {
                    phien: currentSessionId,
                    Xuc_xac_1: d1,
                    Xuc_xac_2: d2,
                    Xuc_xac_3: d3,
                    Tong: total,
                    Ket_qua: result,
                    du_doan: lastPrediction ? lastPrediction.ket_qua : null,
                    loai_cau: lastPrediction ? lastPrediction.loai_cau : null,
                    do_tin_cay: lastPrediction ? lastPrediction.do_tin_cay : null,
                    thoi_gian: new Date().toISOString()
                };
                saveHistory(historyEntry);

                // Tạo mảng history cho analyzer
                const historyForAnalyzer = resultHistory.map(h => ({
                    score: h.Tong,
                    Ket_qua: h.Ket_qua,
                    Xuc_xac_1: h.Xuc_xac_1,
                    Xuc_xac_2: h.Xuc_xac_2,
                    Xuc_xac_3: h.Xuc_xac_3
                }));

                // Dự đoán cho phiên tiếp theo (KHÔNG dùng kết quả phiên trước)
                const analysis = smartEngine.analyzeComplexPatterns(historyForAnalyzer);
                
                let finalPrediction = analysis.prediction;
                let finalConfidence = analysis.confidence;
                let finalReason = analysis.reason;
                let finalPattern = '';
                
                // Đảm bảo confidence không dưới 48%
                finalConfidence = Math.max(finalConfidence, 0.48);
                
                // Điều chỉnh theo chiến thuật
                finalConfidence = smartEngine.getAdjustedConfidence(finalConfidence);
                
                // Xử lý trường hợp không có dự đoán (đang thu thập dữ liệu)
                if (!finalPrediction) {
                    // Dùng phân tích thống kê đơn giản
                    const last20 = resultHistory.slice(-20);
                    const taiCount = last20.filter(h => h.Ket_qua === 'Tài').length;
                    finalPrediction = taiCount > 10 ? 'Xỉu' : 'Tài';
                    finalConfidence = 0.5;
                    finalReason = 'Đang thu thập dữ liệu, dùng thống kê cơ bản';
                }
                
                // Xác định loại cầu dựa trên phân tích
                let patternType = 'Phân tích tổng hợp';
                if (analysis.reason && analysis.reason.includes('chu kỳ')) patternType = 'Cầu chu kỳ';
                else if (analysis.reason && analysis.reason.includes('nhịp điệu')) patternType = 'Cầu nhịp điệu';
                else if (analysis.reason && analysis.reason.includes('ma trận')) patternType = 'Cầu chuyển tiếp';
                else if (analysis.reason && analysis.reason.includes('đối xứng')) patternType = 'Cầu đối xứng';
                else if (analysis.reason && analysis.reason.includes('Fibonacci')) patternType = 'Cầu Fibonacci';
                else patternType = 'Phân tích đa chiều';
                
                // Xử lý thua liên tiếp - chống đảo
                if (stats.consecutiveLosses >= 3) {
                    finalPrediction = finalPrediction === 'Tài' ? 'Xỉu' : 'Tài';
                    finalConfidence = 0.48;
                    patternType = 'CHỐNG ĐẢO (SAU ' + stats.consecutiveLosses + ' LẦN THUA)';
                    finalReason = 'Chống đảo do thua liên tiếp';
                }

                // Lưu dự đoán cho phiên tiếp theo
                lastPrediction = {
                    phien: currentSessionId ? parseInt(currentSessionId) + 1 : null,
                    ket_qua: finalPrediction,
                    loai_cau: patternType,
                    mau_cau: finalPattern,
                    do_tin_cay: (finalConfidence * 100).toFixed(0) + '%'
                };

                // Trạng thái
                const trangThai = stats.consecutiveLosses >= 3 ? 'Chống đảo' :
                                 (patternType.includes('chu kỳ') ? 'Theo chu kỳ' : 'Phân tích thông minh');

                // Tỉ lệ
                const tiLe = stats.total > 0 ? ((stats.correct / stats.total) * 100).toFixed(1) + '%' : '0%';

                // Cập nhật API
                apiResponseData = {
                    "Phien": currentSessionId,
                    "Xuc_xac_1": d1,
                    "Xuc_xac_2": d2,
                    "Xuc_xac_3": d3,
                    "Tong": total,
                    "Ket_qua": result,
                    "Phien_hien_tai": currentSessionId ? parseInt(currentSessionId) + 1 : null,
                    "Du_doan": finalPrediction,
                    "Loai_cau": patternType,
                    "Mau_cau_phat_hien": finalPattern || (analysis.reason ? analysis.reason.substring(0, 50) : ''),
                    "Do_tin_cay": (finalConfidence * 100).toFixed(0) + '%',
                    "Trang_thai": trangThai,
                    "Ket_qua_du_doan": predictionCorrect ? '✅' : (stats.total > 0 ? '❌' : ''),
                    "Thong_ke": {
                        "tong": stats.total,
                        "dung": stats.correct,
                        "sai": stats.wrong,
                        "ti_le": tiLe
                    },
                    "id": "@tranhoang2286"
                };

                // Log chi tiết
                console.log('\n' + '🟦'.repeat(25));
                console.log(`🎲 Phiên ${apiResponseData.Phien} | KQ: ${result} | Tổng: ${total}`);
                console.log(`📊 Lịch sử: ${historyForAnalyzer.slice(-12).map(h => h.Ket_qua).join(' ')}`);
                console.log(`🔍 Phân tích: ${finalReason || 'Đang phân tích...'}`);
                console.log(`🎯 Dự đoán phiên ${apiResponseData.Phien_hien_tai}: ${finalPrediction} (${(finalConfidence * 100).toFixed(0)}%)`);
                console.log(`📈 Thống kê: Đúng ${stats.correct}/${stats.total} (${tiLe}) ${apiResponseData.Ket_qua_du_doan}`);
                console.log(`🎯 Chiến thuật: ${predictionModel.currentStrategy} | LR: ${smartEngine.learningRate.toFixed(3)}`);
                if (stats.consecutiveLosses > 0) {
                    console.log(`⚠️ Thua liên tiếp: ${stats.consecutiveLosses}`);
                }
                console.log('🟦'.repeat(25) + '\n');

                lastResult = result;
                currentSessionId = null;
            }
        } catch (e) {
            console.error('[❌] Lỗi xử lý message:', e.message);
        }
    });

    ws.on('close', (code, reason) => {
        console.log(`[🔌] WebSocket closed. Code: ${code}`);
        clearInterval(pingInterval);
        clearTimeout(reconnectTimeout);
        reconnectTimeout = setTimeout(connectWebSocket, RECONNECT_DELAY);
    });

    ws.on('error', (err) => {
        console.error('[❌] WebSocket error:', err.message);
        ws.close();
    });
}

// ==================== EXPRESS API ====================
app.get('/api/ditmemaysun', (req, res) => {
    res.json(apiResponseData);
});

app.get('/api/his', (req, res) => {
    const recent = resultHistory.slice(-20).reverse();
    
    res.json({
        success: true,
        total: resultHistory.length,
        data: recent,
        stats: {
            tong: stats.total,
            dung: stats.correct,
            sai: stats.wrong,
            ti_le: stats.total > 0 ? ((stats.correct / stats.total) * 100).toFixed(1) + '%' : '0%',
            consecutive_losses: stats.consecutiveLosses
        },
        strategy: {
            current: predictionModel.currentStrategy,
            performance: stats.strategyPerformance
        }
    });
});

app.get('/api/analysis', (req, res) => {
    const historyForAnalyzer = resultHistory.slice(-30).map(h => ({
        score: h.Tong,
        Ket_qua: h.Ket_qua
    }));
    const analysis = smartEngine.analyzeComplexPatterns(historyForAnalyzer);
    res.json({
        analysis: analysis,
        strategy: predictionModel.currentStrategy,
        recentAccuracy: predictionModel.recentPredictions.slice(-20).filter(p => p.correct).length / 
                        Math.min(20, predictionModel.recentPredictions.length) * 100 + '%'
    });
});

app.get('/', (req, res) => {
    res.json(apiResponseData);
});

app.listen(PORT, () => {
    console.log(`[🌐] Server running at http://localhost:${PORT}`);
    console.log(`[🔄] Proxy rotation enabled: ${proxyList.length} proxies`);
    console.log(`[🤖] Smart prediction engine ready`);
    console.log(`[📊] Min confidence: 48%`);
    console.log(`[🆔] ID: @tranhoang2286`);
});

// ==================== START ====================
connectWebSocket();