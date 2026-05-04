const WebSocket = require('ws');
const express = require('express');
const cors = require('cors');
const fs = require('fs');

const app = express();
app.use(cors());
const PORT = process.env.PORT || 3001;

// ==================== FILE STORAGE ====================
const HISTORY_FILE = './history.json';
const PATTERNS_FILE = './patterns.json';
const MODEL_WEIGHTS_FILE = './model_weights.json';

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

// Load model weights if exists
let modelWeights = {
    'model1': 1.0, 'model2': 1.0, 'model3': 1.0, 'model4': 1.0,
    'model5': 1.0, 'model6': 1.0, 'model7': 1.0, 'model8': 1.0,
    'model9': 1.0, 'model10': 1.0, 'model11': 1.0, 'model12': 1.0,
    'model13': 1.0, 'model14': 1.0, 'model15': 1.0, 'model16': 1.0,
    'model17': 1.0, 'model18': 1.0, 'model19': 1.0, 'model20': 1.0,
    'model21': 1.0
};

// Load sub model weights
let subModelWeights = {};
for (let i = 1; i <= 42; i++) {
    subModelWeights[`sub_model_${i}`] = 1.0;
}

// Load mini model weights
let miniModelWeights = {};
for (let i = 1; i <= 21; i++) {
    miniModelWeights[`mini_model_${i}`] = 1.0;
}

// NÂNG CẤP: Thêm neural network weights
let neuralWeights = {
    // Layer 1: Input -> Hidden (84 models -> 42 neurons)
    w1: [],
    b1: [],
    // Layer 2: Hidden -> Output (42 -> 2 neurons: Tài/Xỉu)
    w2: [],
    b2: [],
    // Hệ số học
    learningRate: 0.01,
    momentum: 0.9,
    lastGradientW1: null,
    lastGradientB1: null,
    lastGradientW2: null,
    lastGradientB2: null
};

// Khởi tạo neural weights
function initNeuralWeights() {
    // Xavier initialization
    const inputSize = 84; // 84 models
    const hiddenSize = 42;
    const outputSize = 2;
    
    const w1Scale = Math.sqrt(2.0 / inputSize);
    const w2Scale = Math.sqrt(2.0 / hiddenSize);
    
    neuralWeights.w1 = [];
    neuralWeights.b1 = [];
    neuralWeights.w2 = [];
    neuralWeights.b2 = [];
    
    for (let i = 0; i < hiddenSize; i++) {
        neuralWeights.w1[i] = [];
        for (let j = 0; j < inputSize; j++) {
            neuralWeights.w1[i][j] = (Math.random() * 2 - 1) * w1Scale;
        }
        neuralWeights.b1[i] = (Math.random() * 2 - 1) * w1Scale;
    }
    
    for (let i = 0; i < outputSize; i++) {
        neuralWeights.w2[i] = [];
        for (let j = 0; j < hiddenSize; j++) {
            neuralWeights.w2[i][j] = (Math.random() * 2 - 1) * w2Scale;
        }
        neuralWeights.b2[i] = (Math.random() * 2 - 1) * w2Scale;
    }
    
    neuralWeights.lastGradientW1 = null;
    neuralWeights.lastGradientB1 = null;
    neuralWeights.lastGradientW2 = null;
    neuralWeights.lastGradientB2 = null;
    
    console.log('[🧠] Neural network đã được khởi tạo (84->42->2)');
}

// Neural network activation functions
function relu(x) {
    return Math.max(0, x);
}

function reluDerivative(x) {
    return x > 0 ? 1 : 0;
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expSum = logits.reduce((sum, l) => sum + Math.exp(l - maxLogit), 0);
    return logits.map(l => Math.exp(l - maxLogit) / expSum);
}

function crossEntropyLoss(predicted, target) {
    return -Math.log(Math.max(predicted[target], 1e-10));
}

// Forward pass qua neural network
function neuralForward(input) {
    // Input -> Hidden
    const hidden = [];
    for (let i = 0; i < 42; i++) {
        let sum = neuralWeights.b1[i];
        for (let j = 0; j < 84; j++) {
            sum += neuralWeights.w1[i][j] * input[j];
        }
        hidden.push(relu(sum));
    }
    
    // Hidden -> Output
    const output = [];
    for (let i = 0; i < 2; i++) {
        let sum = neuralWeights.b2[i];
        for (let j = 0; j < 42; j++) {
            sum += neuralWeights.w2[i][j] * hidden[j];
        }
        output.push(sum);
    }
    
    // Softmax
    const probs = softmax(output);
    
    return {
        hidden: hidden,
        output: output,
        probs: probs,
        prediction: probs[0] > probs[1] ? 'Tài' : 'Xỉu',
        confidence: Math.max(probs[0], probs[1])
    };
}

// Backward pass (training)
function neuralBackward(input, target, neuralOutput) {
    const outputSize = 2;
    const hiddenSize = 42;
    const inputSize = 84;
    
    // Output layer gradient
    const outputGradient = [...neuralOutput.probs];
    outputGradient[target] -= 1; // derivative of cross-entropy with softmax
    
    // Hidden -> Output gradient
    const w2Gradient = [];
    const b2Gradient = [];
    for (let i = 0; i < outputSize; i++) {
        w2Gradient[i] = [];
        for (let j = 0; j < hiddenSize; j++) {
            w2Gradient[i][j] = outputGradient[i] * neuralOutput.hidden[j];
        }
        b2Gradient[i] = outputGradient[i];
    }
    
    // Hidden layer gradient
    const hiddenGradient = [];
    for (let j = 0; j < hiddenSize; j++) {
        let grad = 0;
        for (let i = 0; i < outputSize; i++) {
            grad += outputGradient[i] * neuralWeights.w2[i][j];
        }
        hiddenGradient.push(grad * reluDerivative(neuralOutput.hidden[j]));
    }
    
    // Input -> Hidden gradient
    const w1Gradient = [];
    const b1Gradient = [];
    for (let i = 0; i < hiddenSize; i++) {
        w1Gradient[i] = [];
        for (let j = 0; j < inputSize; j++) {
            w1Gradient[i][j] = hiddenGradient[i] * input[j];
        }
        b1Gradient[i] = hiddenGradient[i];
    }
    
    // Update weights với momentum
    const lr = neuralWeights.learningRate;
    const momentum = neuralWeights.momentum;
    
    // Update w2, b2
    for (let i = 0; i < outputSize; i++) {
        for (let j = 0; j < hiddenSize; j++) {
            const prevGrad = neuralWeights.lastGradientW2 ? 
                (neuralWeights.lastGradientW2[i]?.[j] || 0) : 0;
            const update = lr * w2Gradient[i][j] + momentum * prevGrad;
            neuralWeights.w2[i][j] -= update;
            if (!neuralWeights.lastGradientW2) neuralWeights.lastGradientW2 = [];
            if (!neuralWeights.lastGradientW2[i]) neuralWeights.lastGradientW2[i] = [];
            neuralWeights.lastGradientW2[i][j] = update;
        }
        const prevBiasGrad = neuralWeights.lastGradientB2 ? 
            (neuralWeights.lastGradientB2[i] || 0) : 0;
        const biasUpdate = lr * b2Gradient[i] + momentum * prevBiasGrad;
        neuralWeights.b2[i] -= biasUpdate;
        if (!neuralWeights.lastGradientB2) neuralWeights.lastGradientB2 = [];
        neuralWeights.lastGradientB2[i] = biasUpdate;
    }
    
    // Update w1, b1
    for (let i = 0; i < hiddenSize; i++) {
        for (let j = 0; j < inputSize; j++) {
            const prevGrad = neuralWeights.lastGradientW1 ? 
                (neuralWeights.lastGradientW1[i]?.[j] || 0) : 0;
            const update = lr * w1Gradient[i][j] + momentum * prevGrad;
            neuralWeights.w1[i][j] -= update;
            if (!neuralWeights.lastGradientW1) neuralWeights.lastGradientW1 = [];
            if (!neuralWeights.lastGradientW1[i]) neuralWeights.lastGradientW1[i] = [];
            neuralWeights.lastGradientW1[i][j] = update;
        }
        const prevBiasGrad = neuralWeights.lastGradientB1 ? 
            (neuralWeights.lastGradientB1[i] || 0) : 0;
        const biasUpdate = lr * b1Gradient[i] + momentum * prevBiasGrad;
        neuralWeights.b1[i] -= biasUpdate;
        if (!neuralWeights.lastGradientB1) neuralWeights.lastGradientB1 = [];
        neuralWeights.lastGradientB1[i] = biasUpdate;
    }
    
    // Decay learning rate
    neuralWeights.learningRate *= 0.9999;
    neuralWeights.learningRate = Math.max(neuralWeights.learningRate, 0.001);
}

if (fs.existsSync(MODEL_WEIGHTS_FILE)) {
    try {
        const savedWeights = JSON.parse(fs.readFileSync(MODEL_WEIGHTS_FILE, 'utf8'));
        modelWeights = savedWeights.modelWeights || modelWeights;
        subModelWeights = savedWeights.subModelWeights || subModelWeights;
        miniModelWeights = savedWeights.miniModelWeights || miniModelWeights;
        
        // NÂNG CẤP: Load neural weights nếu có
        if (savedWeights.neuralWeights) {
            neuralWeights = savedWeights.neuralWeights;
            console.log('[🧠] Đã tải neural network weights');
        } else {
            initNeuralWeights();
        }
        
        console.log('[📂] Đã tải model_weights.json');
    } catch (e) {
        console.error('[❌] Lỗi đọc model_weights.json:', e.message);
        initNeuralWeights();
    }
} else {
    initNeuralWeights();
}

// Save history
function saveHistory(entry) {
    resultHistory.push(entry);
    if (resultHistory.length > 1000) resultHistory.shift();
    fs.writeFileSync(HISTORY_FILE, JSON.stringify(resultHistory, null, 2));
}

// NÂNG CẤP: Save model weights với neural weights
function saveModelWeights() {
    const weights = {
        modelWeights,
        subModelWeights,
        miniModelWeights,
        neuralWeights: neuralWeights
    };
    fs.writeFileSync(MODEL_WEIGHTS_FILE, JSON.stringify(weights, null, 2));
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
    // NÂNG CẤP: Thêm thống kê neural
    neuralCorrect: 0,
    neuralTotal: 0,
    neuralAccuracy: 0
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
        "ti_le": "0%",
        "neural_ti_le": "0%"
    },
    "id": "@tranhoang2286"
};

// ==================== TAI XIU ANALYZER NÂNG CẤP ====================
class TaiXiuAnalyzer {
    constructor() {
        // Model weights
        this.modelWeights = modelWeights;
        this.subModelWeights = subModelWeights;
        this.miniModelWeights = miniModelWeights;
        
        // Sub models (42 cái với chuyên môn riêng)
        this.subModels = {};
        this.initSubModels();
        
        // Mini models (21 cái)
        this.miniModels = {};
        this.initMiniModels();
        
        this.performanceHistory = {};
        this.patternLibrary = this.loadPatternLibrary();
        
        // NÂNG CẤP: Thêm bộ nhớ dự đoán gần đây để neural học
        this.recentPredictions = [];
        this.maxRecentPredictions = 50;
    }
    
    loadPatternLibrary() {
        // Thư viện các mẫu cầu đã gặp
        if (fs.existsSync(PATTERNS_FILE)) {
            try {
                return JSON.parse(fs.readFileSync(PATTERNS_FILE, 'utf8'));
            } catch (e) {
                console.error('[❌] Lỗi đọc patterns.json:', e.message);
            }
        }
        return {
            '1-1': [], '2-2': [], '3-3': [], '1-2': [], '2-1': [],
            '2-1-2': [], '1-2-1': [], 'bệt': [], 'loạn': []
        };
    }
    
    savePatternLibrary() {
        fs.writeFileSync(PATTERNS_FILE, JSON.stringify(this.patternLibrary, null, 2));
    }
    
    initSubModels() {
        // 42 sub models với chuyên môn khác nhau
        const subModelSpecialties = {
            // Model 1-6: Chuyên phân tích cầu 1-1 các biến thể
            1: { name: '1-1 thuần', type: '1-1', logic: 'pure', minLength: 4, threshold: 0.9 },
            2: { name: '1-1 biến thể', type: '1-1', logic: 'variant', minLength: 5, threshold: 0.8 },
            3: { name: '1-1 dài hạn', type: '1-1', logic: 'long', minLength: 8, threshold: 0.75 },
            4: { name: '1-1 kết hợp', type: '1-1', logic: 'hybrid', minLength: 6, threshold: 0.7 },
            5: { name: '1-1 gãy', type: '1-1', logic: 'break', minLength: 6, threshold: 0.8 },
            6: { name: '1-1 phục hồi', type: '1-1', logic: 'recovery', minLength: 7, threshold: 0.7 },
            
            // Model 7-12: Chuyên cầu 2-2
            7: { name: '2-2 chuẩn', type: '2-2', logic: 'pure', minLength: 6, threshold: 0.9 },
            8: { name: '2-2 lệch', type: '2-2', logic: 'offset', minLength: 7, threshold: 0.8 },
            9: { name: '2-2 biến tướng', type: '2-2', logic: 'variant', minLength: 8, threshold: 0.75 },
            10: { name: '2-2 kết hợp 1-1', type: '2-2', logic: 'hybrid', minLength: 8, threshold: 0.7 },
            11: { name: '2-2 dài', type: '2-2', logic: 'long', minLength: 10, threshold: 0.8 },
            12: { name: '2-2 bẻ', type: '2-2', logic: 'break', minLength: 7, threshold: 0.85 },
            
            // Model 13-18: Chuyên cầu bệt
            13: { name: 'bệt ngắn', type: 'bệt', logic: 'short', minLength: 3, threshold: 0.8 },
            14: { name: 'bệt trung', type: 'bệt', logic: 'medium', minLength: 5, threshold: 0.85 },
            15: { name: 'bệt dài', type: 'bệt', logic: 'long', minLength: 7, threshold: 0.9 },
            16: { name: 'bệt gãy', type: 'bệt', logic: 'break', minLength: 5, threshold: 0.8 },
            17: { name: 'bệt xen kẽ', type: 'bệt', logic: 'hybrid', minLength: 6, threshold: 0.7 },
            18: { name: 'siêu bệt', type: 'bệt', logic: 'super', minLength: 10, threshold: 0.95 },
            
            // Model 19-24: Chuyên cầu 3-3
            19: { name: '3-3 chuẩn', type: '3-3', logic: 'pure', minLength: 9, threshold: 0.9 },
            20: { name: '3-3 biến thể', type: '3-3', logic: 'variant', minLength: 10, threshold: 0.8 },
            21: { name: '3-3 ngắn', type: '3-3', logic: 'short', minLength: 6, threshold: 0.7 },
            22: { name: '3-3 kết hợp', type: '3-3', logic: 'hybrid', minLength: 9, threshold: 0.75 },
            23: { name: '3-3 bẻ', type: '3-3', logic: 'break', minLength: 8, threshold: 0.8 },
            24: { name: '3-3 dài', type: '3-3', logic: 'long', minLength: 12, threshold: 0.85 },
            
            // Model 25-30: Chuyên cầu 2-1-2 và 1-2-1
            25: { name: '2-1-2 chuẩn', type: '2-1-2', logic: 'pure', minLength: 5, threshold: 0.9 },
            26: { name: '2-1-2 biến thể', type: '2-1-2', logic: 'variant', minLength: 6, threshold: 0.8 },
            27: { name: '2-1-2 dài', type: '2-1-2', logic: 'long', minLength: 8, threshold: 0.8 },
            28: { name: '1-2-1 chuẩn', type: '1-2-1', logic: 'pure', minLength: 5, threshold: 0.9 },
            29: { name: '1-2-1 biến thể', type: '1-2-1', logic: 'variant', minLength: 6, threshold: 0.8 },
            30: { name: '1-2-1 dài', type: '1-2-1', logic: 'long', minLength: 8, threshold: 0.8 },
            
            // Model 31-36: Chuyên bẻ cầu và chuyển tiếp
            31: { name: 'bẻ cầu 1-1', type: 'break', logic: 'break11', minLength: 4, threshold: 0.85 },
            32: { name: 'bẻ cầu 2-2', type: 'break', logic: 'break22', minLength: 5, threshold: 0.85 },
            33: { name: 'bẻ cầu bệt', type: 'break', logic: 'breakStreak', minLength: 4, threshold: 0.8 },
            34: { name: 'chuyển tiếp 1-1 sang 2-2', type: 'transition', logic: '11to22', minLength: 6, threshold: 0.75 },
            35: { name: 'chuyển tiếp 2-2 sang 1-1', type: 'transition', logic: '22to11', minLength: 6, threshold: 0.75 },
            36: { name: 'chuyển tiếp bệt sang 1-1', type: 'transition', logic: 'streakTo11', minLength: 5, threshold: 0.7 },
            
            // Model 37-42: Chuyên phân tích tổng hợp
            37: { name: 'phân tích tần suất', type: 'frequency', logic: 'frequency', minLength: 10, threshold: 0.7 },
            38: { name: 'phân tích chu kỳ', type: 'cycle', logic: 'cycle', minLength: 12, threshold: 0.7 },
            39: { name: 'phân tích đối xứng', type: 'symmetry', logic: 'symmetry', minLength: 8, threshold: 0.75 },
            40: { name: 'phân tích Fibonacci', type: 'fibonacci', logic: 'fibonacci', minLength: 8, threshold: 0.7 },
            41: { name: 'phân tích xu hướng dài', type: 'trend', logic: 'longTrend', minLength: 15, threshold: 0.8 },
            42: { name: 'tổng hợp siêu cầu', type: 'super', logic: 'super', minLength: 20, threshold: 0.85 }
        };
        
        for (let i = 1; i <= 42; i++) {
            this.subModels[`sub_model_${i}`] = {
                ...subModelSpecialties[i],
                weight: this.subModelWeights[`sub_model_${i}`] || 1.0,
                accuracy: 0.5,
                predictions: []
            };
        }
    }
    
    initMiniModels() {
        const specialties = {
            1: 'phat_hien_cau_dep',
            2: 'du_doan_bien_dong',
            3: 'phan_tich_so_sanh',
            4: 'nhan_dien_xu_huong_cuc_bo',
            5: 'tinh_toan_xac_suat_cao',
            6: 'phat_hien_diem_gay',
            7: 'du_doan_nguong',
            8: 'phan_tich_chuoi',
            9: 'nhan_dien_mau_lap',
            10: 'tinh_he_so_tuong_quan',
            11: 'du_doan_doan_nhiet',
            12: 'phan_tich_pha',
            13: 'nhan_dien_song',
            14: 'tinh_toan_momentum',
            15: 'du_doan_hoi_phuc',
            16: 'phat_hien_dot_bien',
            17: 'phan_tich_can_bang',
            18: 'nhan_dien_tan_so',
            19: 'du_doan_chu_ky',
            20: 'tinh_toan_ma_tran',
            21: 'phan_tich_tong_hop'
        };
        
        for (let i = 1; i <= 21; i++) {
            this.miniModels[`mini_model_${i}`] = {
                weight: this.miniModelWeights[`mini_model_${i}`] || 1.0,
                accuracy: 0.5,
                specialty: specialties[i] || 'chung',
                predictions: []
            };
        }
    }
    
    // Helper: lấy mảng kết quả từ history
    getResultArray(history) {
        return history.map(h => h.Ket_qua || (h.score >= 11 ? 'Tài' : 'Xỉu'));
    }
    
    // NÂNG CẤP: Lấy feature vector cho neural network (84 features)
    getNeuralInput(history) {
        const features = [];
        
        // 21 main model features
        const mainResults = this.getAllMainModelResults(history);
        for (let i = 1; i <= 21; i++) {
            const result = mainResults[`model${i}`];
            if (result && result.prediction) {
                features.push(result.prediction === 'Tài' ? result.confidence : -result.confidence);
            } else {
                features.push(0);
            }
        }
        
        // 42 sub model features
        const subResults = this.getAllSubModelResults(history);
        for (let i = 1; i <= 42; i++) {
            const result = subResults[`sub_model_${i}`];
            if (result && result.prediction) {
                features.push(result.prediction === 'Tài' ? result.confidence : -result.confidence);
            } else {
                features.push(0);
            }
        }
        
        // 21 mini model features
        const miniResults = this.getAllMiniModelResults(history);
        for (let i = 1; i <= 21; i++) {
            const result = miniResults[`mini_model_${i}`];
            if (result && result.prediction) {
                features.push(result.prediction === 'Tài' ? result.confidence : -result.confidence);
            } else {
                features.push(0);
            }
        }
        
        return features;
    }
    
    // Lấy tất cả kết quả main models
    getAllMainModelResults(history) {
        const results = {};
        results.model1 = this.analyzeBasicPatterns(history);
        results.model2 = this.analyzeTrend(history);
        results.model3 = this.analyzeImbalance(history);
        results.model4 = this.analyzeShortTerm(history);
        results.model5 = this.analyzeLongTermPattern(history);
        results.model6 = this.analyzeDiceSequence(history);
        results.model7 = this.analyzeMomentum(history);
        results.model8 = this.analyzeReversal(history);
        results.model9 = this.analyzeVolatility(history);
        results.model10 = this.analyzeHarmonic(history);
        results.model11 = this.analyzeDiceVolatility(history);
        results.model12 = this.analyzePatternMemory(history);
        results.model13 = this.analyzeFibonacciLevels(history);
        results.model14 = this.analyzeSupportResistance(history);
        results.model15 = this.analyzeBreakout(history);
        results.model16 = this.analyzeConsolidation(history);
        results.model17 = this.analyzeWavePattern(history);
        results.model18 = this.analyzeCorrelation(history);
        results.model19 = this.analyzeSeasonality(history);
        results.model20 = this.analyzeSentiment(history);
        results.model21 = this.analyzeEnsembleBasic(history);
        return results;
    }
    
    // Lấy tất cả kết quả sub models
    getAllSubModelResults(history) {
        const results = {};
        for (let i = 1; i <= 42; i++) {
            results[`sub_model_${i}`] = this.runSubModel(i, history);
        }
        return results;
    }
    
    // Lấy tất cả kết quả mini models
    getAllMiniModelResults(history) {
        const results = {};
        for (let i = 1; i <= 21; i++) {
            results[`mini_model_${i}`] = this.runMiniModel(i, history);
        }
        return results;
    }
    
    // NÂNG CẤP: Thêm các model mới (model 5-21 mở rộng)
    
    // Model 5: Long-term pattern analysis
    analyzeLongTermPattern(history) {
        if (history.length < 20) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu dài hạn' };
        }
        const results = this.getResultArray(history);
        const last20 = results.slice(-20);
        
        // Phân tích tỉ lệ Tài/Xỉu trong 20 phiên
        const taiCount = last20.filter(r => r === 'Tài').length;
        const xiuCount = 20 - taiCount;
        
        // Tìm pattern dài hạn
        let prediction, confidence, reason;
        
        if (taiCount >= 13) {
            prediction = 'Xỉu';
            confidence = 0.65;
            reason = 'Tài áp đảo dài hạn (≥65%), dự đoán Xỉu cân bằng';
        } else if (xiuCount >= 13) {
            prediction = 'Tài';
            confidence = 0.65;
            reason = 'Xỉu áp đảo dài hạn (≥65%), dự đoán Tài cân bằng';
        } else if (taiCount >= 11) {
            prediction = 'Tài';
            confidence = 0.55;
            reason = 'Xu hướng Tài nhẹ trong dài hạn';
        } else if (xiuCount >= 11) {
            prediction = 'Xỉu';
            confidence = 0.55;
            reason = 'Xu hướng Xỉu nhẹ trong dài hạn';
        } else {
            prediction = results[results.length - 1];
            confidence = 0.5;
            reason = 'Cân bằng dài hạn';
        }
        
        return { prediction, confidence, reason, tai_count: taiCount, xiu_count: xiuCount };
    }
    
    // Model 6: Phân tích chuỗi xúc xắc
    analyzeDiceSequence(history) {
        if (history.length < 5) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu xúc xắc' };
        }
        
        const recentHistory = history.slice(-10);
        const dicePatterns = [];
        
        recentHistory.forEach(h => {
            if (h.Xuc_xac_1 && h.Xuc_xac_2 && h.Xuc_xac_3) {
                dicePatterns.push({
                    dice: [h.Xuc_xac_1, h.Xuc_xac_2, h.Xuc_xac_3].sort(),
                    result: h.Ket_qua || (h.score >= 11 ? 'Tài' : 'Xỉu')
                });
            }
        });
        
        if (dicePatterns.length < 3) {
            return { prediction: null, confidence: 0, reason: 'Không đủ pattern xúc xắc' };
        }
        
        // Tìm pattern xúc xắc lặp lại
        const lastDice = dicePatterns[dicePatterns.length - 1].dice.join(',');
        const prevDice = dicePatterns.slice(0, -1).map(d => d.dice.join(','));
        
        let matchCount = 0;
        let matchResults = [];
        
        prevDice.forEach((d, i) => {
            if (d === lastDice) {
                matchCount++;
                if (i + 1 < dicePatterns.length) {
                    matchResults.push(dicePatterns[i + 1].result);
                }
            }
        });
        
        if (matchCount > 0 && matchResults.length > 0) {
            const taiMatches = matchResults.filter(r => r === 'Tài').length;
            const prediction = taiMatches > matchResults.length / 2 ? 'Tài' : 'Xỉu';
            const confidence = 0.5 + (matchCount * 0.05);
            
            return {
                prediction,
                confidence: Math.min(confidence, 0.75),
                reason: `Pattern xúc xắc ${lastDice} xuất hiện ${matchCount} lần, kết quả tiếp theo thường là ${prediction}`
            };
        }
        
        return {
            prediction: dicePatterns[dicePatterns.length - 1].result,
            confidence: 0.4,
            reason: 'Không tìm thấy pattern xúc xắc lặp'
        };
    }
    
    // Model 7: Momentum analysis
    analyzeMomentum(history) {
        if (history.length < 6) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu momentum' };
        }
        
        const results = this.getResultArray(history);
        const last = results[results.length - 1];
        const last6 = results.slice(-6);
        
        // Tính momentum
        let momentum = 0;
        for (let i = 0; i < last6.length; i++) {
            momentum += last6[i] === 'Tài' ? 1 : -1;
        }
        
        const absMomentum = Math.abs(momentum);
        
        if (absMomentum >= 4) {
            // Momentum mạnh
            const direction = momentum > 0 ? 'Tài' : 'Xỉu';
            return {
                prediction: direction,
                confidence: 0.65 + (absMomentum * 0.05),
                momentum_value: momentum,
                reason: `Momentum ${direction} mạnh (${momentum})`
            };
        } else if (absMomentum >= 2) {
            // Momentum nhẹ
            const direction = momentum > 0 ? 'Tài' : 'Xỉu';
            return {
                prediction: direction,
                confidence: 0.55,
                momentum_value: momentum,
                reason: `Momentum ${direction} nhẹ (${momentum})`
            };
        } else {
            // Không có momentum
            return {
                prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                confidence: 0.5,
                momentum_value: momentum,
                reason: 'Không có momentum rõ ràng'
            };
        }
    }
    
    // Model 8: Reversal detection
    analyzeReversal(history) {
        if (history.length < 4) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu đảo chiều' };
        }
        
        const results = this.getResultArray(history);
        const last4 = results.slice(-4);
        const last = results[results.length - 1];
        const other = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        // Pattern đảo chiều cổ điển
        // Đỉnh đôi: TXT T -> X
        if (last4.length === 4) {
            if (last4[0] === 'Xỉu' && last4[1] === 'Tài' && last4[2] === 'Xỉu' && last4[3] === 'Tài') {
                return {
                    prediction: 'Xỉu',
                    confidence: 0.7,
                    reason: 'Phát hiện đỉnh đôi, dự đoán đảo chiều Xỉu'
                };
            }
            // Đáy đôi: XTX X -> T
            if (last4[0] === 'Tài' && last4[1] === 'Xỉu' && last4[2] === 'Tài' && last4[3] === 'Xỉu') {
                return {
                    prediction: 'Tài',
                    confidence: 0.7,
                    reason: 'Phát hiện đáy đôi, dự đoán đảo chiều Tài'
                };
            }
        }
        
        if (results.length >= 5) {
            const last5 = results.slice(-5);
            // Vai đầu vai đỉnh: T X T X T -> X
            if (last5[0] === 'Tài' && last5[1] === 'Xỉu' && last5[2] === 'Tài' && 
                last5[3] === 'Xỉu' && last5[4] === 'Tài') {
                return {
                    prediction: 'Xỉu',
                    confidence: 0.75,
                    reason: 'Phát hiện vai đầu vai đỉnh, dự đoán đảo chiều Xỉu'
                };
            }
            // Vai đầu vai đáy: X T X T X -> T
            if (last5[0] === 'Xỉu' && last5[1] === 'Tài' && last5[2] === 'Xỉu' && 
                last5[3] === 'Tài' && last5[4] === 'Xỉu') {
                return {
                    prediction: 'Tài',
                    confidence: 0.75,
                    reason: 'Phát hiện vai đầu vai đáy, dự đoán đảo chiều Tài'
                };
            }
        }
        
        return { prediction: null, confidence: 0, reason: 'Không phát hiện đảo chiều' };
    }
    
    // Model 9: Volatility analysis
    analyzeVolatility(history) {
        if (history.length < 10) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu biến động' };
        }
        
        const results = this.getResultArray(history);
        const last10 = results.slice(-10);
        const last = results[results.length - 1];
        const other = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        // Đếm số lần đổi chiều
        let reversals = 0;
        for (let i = 1; i < last10.length; i++) {
            if (last10[i] !== last10[i-1]) reversals++;
        }
        
        const reversalRate = reversals / 9;
        
        if (reversalRate >= 0.7) {
            // Biến động cao - dễ đổi chiều
            return {
                prediction: other,
                confidence: 0.6,
                reversal_rate: reversalRate,
                reason: `Biến động cao (${(reversalRate*100).toFixed(0)}%), dự đoán đổi chiều`
            };
        } else if (reversalRate <= 0.3) {
            // Biến động thấp - dễ bệt
            return {
                prediction: last,
                confidence: 0.6,
                reversal_rate: reversalRate,
                reason: `Biến động thấp (${(reversalRate*100).toFixed(0)}%), dự đoán tiếp tục bệt`
            };
        } else {
            return {
                prediction: last,
                confidence: 0.5,
                reversal_rate: reversalRate,
                reason: `Biến động trung bình (${(reversalRate*100).toFixed(0)}%)`
            };
        }
    }
    
    // Model 10: Harmonic pattern
    analyzeHarmonic(history) {
        if (history.length < 8) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu harmonic' };
        }
        
        const results = this.getResultArray(history);
        const last8 = results.slice(-8);
        
        // Gartley pattern: T X T T X T X X -> T
        const gartleyBull = ['Tài', 'Xỉu', 'Tài', 'Tài', 'Xỉu', 'Tài', 'Xỉu', 'Xỉu'];
        // Gartley bear: X T X X T X T T -> X
        const gartleyBear = ['Xỉu', 'Tài', 'Xỉu', 'Xỉu', 'Tài', 'Xỉu', 'Tài', 'Tài'];
        
        if (JSON.stringify(last8) === JSON.stringify(gartleyBull)) {
            return { prediction: 'Tài', confidence: 0.8, reason: 'Phát hiện Gartley Bull pattern' };
        }
        if (JSON.stringify(last8) === JSON.stringify(gartleyBear)) {
            return { prediction: 'Xỉu', confidence: 0.8, reason: 'Phát hiện Gartley Bear pattern' };
        }
        
        // Kiểm tra gần đúng (6/8 khớp)
        let bullMatch = 0, bearMatch = 0;
        for (let i = 0; i < 8; i++) {
            if (last8[i] === gartleyBull[i]) bullMatch++;
            if (last8[i] === gartleyBear[i]) bearMatch++;
        }
        
        if (bullMatch >= 6) {
            return { prediction: 'Tài', confidence: 0.6 + (bullMatch-6)*0.05, reason: `Gartley Bull gần đúng (${bullMatch}/8)` };
        }
        if (bearMatch >= 6) {
            return { prediction: 'Xỉu', confidence: 0.6 + (bearMatch-6)*0.05, reason: `Gartley Bear gần đúng (${bearMatch}/8)` };
        }
        
        return { prediction: null, confidence: 0, reason: 'Không phát hiện harmonic' };
    }
    
    // Model 12: Pattern memory - học từ quá khứ
    analyzePatternMemory(history) {
        if (history.length < 6) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu pattern memory' };
        }
        
        const results = this.getResultArray(history);
        const last5 = results.slice(-5).join('');
        const last = results[results.length - 1];
        
        // Tìm trong pattern library
        for (let [type, patterns] of Object.entries(this.patternLibrary)) {
            if (patterns.includes(last5)) {
                // Tìm kết quả tiếp theo từ lịch sử
                const nextResults = [];
                for (let i = 0; i < results.length - 5; i++) {
                    if (results.slice(i, i+5).join('') === last5) {
                        nextResults.push(results[i+5] || null);
                    }
                }
                
                const validNext = nextResults.filter(r => r !== null);
                if (validNext.length > 0) {
                    const taiCount = validNext.filter(r => r === 'Tài').length;
                    const prediction = taiCount > validNext.length / 2 ? 'Tài' : 'Xỉu';
                    const confidence = 0.6 + (Math.max(taiCount, validNext.length - taiCount) / validNext.length) * 0.2;
                    
                    return {
                        prediction,
                        confidence: Math.min(confidence, 0.85),
                        reason: `Pattern ${last5} (${type}) đã xuất hiện ${validNext.length} lần, tiếp theo thường là ${prediction}`
                    };
                }
            }
        }
        
        return { prediction: null, confidence: 0, reason: 'Pattern chưa có trong bộ nhớ' };
    }
    
    // Model 13: Fibonacci levels
    analyzeFibonacciLevels(history) {
        if (history.length < 8) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu Fibonacci' };
        }
        
        const results = this.getResultArray(history);
        const last = results[results.length - 1];
        const other = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        // Fibonacci sequence positions: 1,1,2,3,5,8
        const fibPositions = [1, 2, 3, 5, 8];
        
        for (let fib of fibPositions) {
            if (results.length > fib * 2) {
                const currentSegment = results.slice(-fib);
                const prevSegment = results.slice(-fib*2, -fib);
                
                if (JSON.stringify(currentSegment) === JSON.stringify(prevSegment)) {
                    return {
                        prediction: currentSegment[0],
                        confidence: 0.65 + (fib * 0.02),
                        reason: `Chu kỳ Fibonacci ${fib} phiên lặp lại`
                    };
                }
            }
        }
        
        // Fibonacci retracement: nếu bệt dài, dự đoán retrace
        const streak = this.getStreak(results.slice(0, -1));
        if (streak >= 5) {
            return {
                prediction: other,
                confidence: 0.6,
                reason: `Fibonacci retracement sau ${streak} phiên bệt`
            };
        }
        
        return { prediction: null, confidence: 0, reason: 'Không phát hiện Fibonacci' };
    }
    
    // Model 14: Support/Resistance
    analyzeSupportResistance(history) {
        if (history.length < 12) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu S/R' };
        }
        
        const results = this.getResultArray(history);
        const last = results[results.length - 1];
        const last12 = results.slice(-12);
        
        // Đếm số lần Tài và Xỉu
        let taiCount = 0, xiuCount = 0;
        last12.forEach(r => r === 'Tài' ? taiCount++ : xiuCount++);
        
        // Nếu Tài quá nhiều, Xỉu là support
        if (taiCount >= 8) {
            return {
                prediction: 'Xỉu',
                confidence: 0.55 + (taiCount - 7) * 0.05,
                reason: `Tài vượt ngưỡng kháng cự (${taiCount}/12), dự đoán Xỉu`
            };
        }
        
        if (xiuCount >= 8) {
            return {
                prediction: 'Tài',
                confidence: 0.55 + (xiuCount - 7) * 0.05,
                reason: `Xỉu vượt ngưỡng hỗ trợ (${xiuCount}/12), dự đoán Tài`
            };
        }
        
        return { prediction: null, confidence: 0, reason: 'Chưa chạm ngưỡng S/R' };
    }
    
    // Model 15: Breakout detection
    analyzeBreakout(history) {
        if (history.length < 6) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu breakout' };
        }
        
        const results = this.getResultArray(history);
        const last6 = results.slice(-6);
        const last = results[results.length - 1];
        const other = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        // Phát hiện phá vỡ pattern
        // Nếu 5 phiên xen kẽ rồi phiên thứ 6 giống phiên thứ 5
        if (last6[0] !== last6[1] && last6[1] !== last6[2] && last6[2] !== last6[3] && 
            last6[3] !== last6[4] && last6[4] === last6[5]) {
            return {
                prediction: last,
                confidence: 0.7,
                reason: 'Phát hiện breakout khỏi cầu xen kẽ'
            };
        }
        
        // Nếu 4 phiên bệt rồi đổi
        const streak = this.getStreak(results.slice(-5));
        if (streak >= 4 && last6[4] !== last6[5]) {
            return {
                prediction: last,
                confidence: 0.75,
                reason: `Breakout khỏi bệt ${streak} phiên`
            };
        }
        
        return { prediction: null, confidence: 0, reason: 'Không phát hiện breakout' };
    }
    
    // Model 16: Consolidation
    analyzeConsolidation(history) {
        if (history.length < 8) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu consolidation' };
        }
        
        const results = this.getResultArray(history);
        const last8 = results.slice(-8);
        const last = results[results.length - 1];
        
        // Kiểm tra tích lũy (sideways)
        const uniqueValues = new Set(last8);
        if (uniqueValues.size === 2) {
            const taiCount = last8.filter(r => r === 'Tài').length;
            if (taiCount >= 3 && taiCount <= 5) {
                // Đang tích lũy, dự đoán breakout
                const streak = this.getStreak(results.slice(-3));
                if (streak >= 2) {
                    return {
                        prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                        confidence: 0.6,
                        reason: 'Đang tích lũy, dự đoán phá vỡ'
                    };
                }
                return {
                    prediction: last,
                    confidence: 0.55,
                    reason: 'Đang tích lũy, tiếp tục xu hướng'
                };
            }
        }
        
        return { prediction: null, confidence: 0, reason: 'Không trong vùng tích lũy' };
    }
    
    // Model 17: Wave pattern (Elliott wave simplified)
    analyzeWavePattern(history) {
        if (history.length < 8) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu sóng' };
        }
        
        const results = this.getResultArray(history);
        const last = results[results.length - 1];
        const other = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        // Đếm số lần đổi chiều trong 8 phiên gần nhất
        const last8 = results.slice(-8);
        let waveCount = 0;
        let currentDirection = last8[0];
        
        for (let i = 1; i < last8.length; i++) {
            if (last8[i] !== currentDirection) {
                waveCount++;
                currentDirection = last8[i];
            }
        }
        
        // Elliott wave: 5 waves impulse + 3 waves corrective
        if (waveCount === 4 && last8[7] === last8[0]) {
            // Hoàn thành 5 sóng
            return {
                prediction: other,
                confidence: 0.7,
                reason: 'Hoàn thành 5 sóng Elliott, dự đoán điều chỉnh'
            };
        } else if (waveCount === 2 || waveCount === 3) {
            return {
                prediction: last,
                confidence: 0.55,
                reason: `Đang trong sóng ${waveCount + 1}`
            };
        }
        
        return { prediction: null, confidence: 0, reason: 'Không phát hiện sóng Elliott' };
    }
    
    // Model 18: Correlation với điểm số
    analyzeCorrelation(history) {
        if (history.length < 5) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu correlation' };
        }
        
        const recentHistory = history.slice(-10);
        const scores = recentHistory.map(h => h.Tong || h.score || 0).filter(s => s > 0);
        const results = recentHistory.map(h => h.Ket_qua || (h.score >= 11 ? 'Tài' : 'Xỉu'));
        
        if (scores.length < 3) {
            return { prediction: null, confidence: 0, reason: 'Không đủ điểm số' };
        }
        
        // Tính trung bình điểm
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const lastScore = scores[scores.length - 1];
        
        // Nếu điểm đang cao bất thường
        if (lastScore > avgScore + 3 && lastScore >= 15) {
            return {
                prediction: 'Xỉu',
                confidence: 0.65,
                reason: `Điểm cao bất thường (${lastScore} vs TB ${avgScore.toFixed(1)}), dự đoán Xỉu`
            };
        }
        
        // Nếu điểm đang thấp bất thường
        if (lastScore < avgScore - 3 && lastScore <= 6) {
            return {
                prediction: 'Tài',
                confidence: 0.65,
                reason: `Điểm thấp bất thường (${lastScore} vs TB ${avgScore.toFixed(1)}), dự đoán Tài`
            };
        }
        
        // Xu hướng điểm
        if (scores.length >= 3) {
            const last3Scores = scores.slice(-3);
            if (last3Scores[0] < last3Scores[1] && last3Scores[1] < last3Scores[2]) {
                // Điểm tăng dần
                return {
                    prediction: lastScore >= 10 ? 'Tài' : 'Xỉu',
                    confidence: 0.55,
                    reason: 'Điểm đang tăng dần'
                };
            }
            if (last3Scores[0] > last3Scores[1] && last3Scores[1] > last3Scores[2]) {
                return {
                    prediction: lastScore >= 11 ? 'Tài' : 'Xỉu',
                    confidence: 0.55,
                    reason: 'Điểm đang giảm dần'
                };
            }
        }
        
        return { prediction: null, confidence: 0, reason: 'Không có tương quan đặc biệt' };
    }
    
    // Model 19: Seasonality (theo phiên)
    analyzeSeasonality(history) {
        if (history.length < 10) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu seasonality' };
        }
        
        const results = this.getResultArray(history);
        const last = results[results.length - 1];
        
        // Phân tích theo vị trí phiên (chu kỳ 5)
        const position = history.length % 5;
        const samePosition = [];
        
        for (let i = position; i < history.length - 1; i += 5) {
            if (i < results.length) {
                samePosition.push(results[i]);
            }
        }
        
        if (samePosition.length >= 2) {
            const taiCount = samePosition.filter(r => r === 'Tài').length;
            const xiuCount = samePosition.length - taiCount;
            
            if (taiCount > xiuCount * 1.5) {
                return {
                    prediction: 'Tài',
                    confidence: 0.55 + (samePosition.length * 0.02),
                    reason: `Vị trí #${position+1} trong chu kỳ 5 thường ra Tài (${taiCount}/${samePosition.length})`
                };
            }
            if (xiuCount > taiCount * 1.5) {
                return {
                    prediction: 'Xỉu',
                    confidence: 0.55 + (samePosition.length * 0.02),
                    reason: `Vị trí #${position+1} trong chu kỳ 5 thường ra Xỉu (${xiuCount}/${samePosition.length})`
                };
            }
        }
        
        return { prediction: null, confidence: 0, reason: 'Không có tính mùa vụ rõ ràng' };
    }
    
    // Model 20: Sentiment analysis
    analyzeSentiment(history) {
        if (history.length < 3) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu sentiment' };
        }
        
        const results = this.getResultArray(history);
        const last = results[results.length - 1];
        const other = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        // "Sentiment" dựa trên pattern gần đây
        const last3 = results.slice(-3);
        const streak = this.getStreak(last3);
        
        // Market sentiment
        if (streak === 3) {
            // Quá bullish/bearish -> có thể đảo chiều
            return {
                prediction: other,
                confidence: 0.6,
                reason: `Sentiment cực đoan (${streak} ${last} liên tiếp), dự đoán đảo chiều`
            };
        } else if (streak === 2) {
            return {
                prediction: last,
                confidence: 0.55,
                reason: `Sentiment đang theo ${last}`
            };
        }
        
        // Fear & Greed index đơn giản
        const last10 = results.slice(-10);
        const taiCount = last10.filter(r => r === 'Tài').length;
        
        if (taiCount >= 7) {
            return { prediction: 'Xỉu', confidence: 0.6, reason: 'Greed index cao, dự đoán Xỉu' };
        } else if (taiCount <= 3) {
            return { prediction: 'Tài', confidence: 0.6, reason: 'Fear index cao, dự đoán Tài' };
        }
        
        return { prediction: null, confidence: 0, reason: 'Sentiment trung tính' };
    }
    
    // Model 21: Ensemble cơ bản (meta-model)
    analyzeEnsembleBasic(history) {
        if (history.length < 2) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu ensemble' };
        }
        
        // Lấy dự đoán từ các model 1-4
        const m1 = this.analyzeBasicPatterns(history);
        const m2 = this.analyzeTrend(history);
        const m3 = this.analyzeImbalance(history);
        const m4 = this.analyzeShortTerm(history);
        
        const models = [m1, m2, m3, m4].filter(m => m && m.prediction);
        
        if (models.length === 0) {
            return { prediction: null, confidence: 0, reason: 'Không model nào có dự đoán' };
        }
        
        let taiVotes = 0, xiuVotes = 0;
        models.forEach(m => {
            if (m.prediction === 'Tài') taiVotes += m.confidence;
            else xiuVotes += m.confidence;
        });
        
        const totalVotes = taiVotes + xiuVotes;
        const prediction = taiVotes > xiuVotes ? 'Tài' : 'Xỉu';
        const confidence = Math.max(taiVotes, xiuVotes) / totalVotes;
        
        return {
            prediction,
            confidence,
            reason: `Ensemble 4 models: ${taiVotes.toFixed(2)}T vs ${xiuVotes.toFixed(2)}X`
        };
    }
    
    // ==================== SUB MODELS THÔNG MINH ====================
    
    // Model 1-6: Chuyên cầu 1-1
    runSubModel11(results, model) {
        if (results.length < model.minLength) return null;
        
        const last = results[results.length - 1];
        const last4 = results.slice(-4);
        const last6 = results.slice(-6);
        
        switch (model.logic) {
            case 'pure':
                if (this.isPerfectAlternating(results, 4)) {
                    return {
                        prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                        confidence: 0.9,
                        reason: 'Phát hiện cầu 1-1 thuần túy'
                    };
                }
                break;
                
            case 'variant':
                if (this.isAlternatingWithTolerance(results, 1)) {
                    return {
                        prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                        confidence: 0.8,
                        reason: 'Phát hiện cầu 1-1 biến thể'
                    };
                }
                break;
                
            case 'long':
                const longResults = results.slice(-12);
                const altCount = this.countAlternating(longResults);
                if (altCount >= 8) {
                    return {
                        prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                        confidence: 0.7 + (altCount / 20),
                        reason: `Cầu 1-1 dài hạn với ${altCount}/11 cặp xen kẽ`
                    };
                }
                break;
                
            case 'hybrid':
                const recent = results.slice(-5);
                if (recent[0] !== recent[1] && recent[1] !== recent[2] && recent[3] !== recent[4]) {
                    return {
                        prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                        confidence: 0.7,
                        reason: 'Phát hiện cầu 1-1 kết hợp'
                    };
                }
                break;
                
            case 'break':
                if (last4[0] !== last4[1] && last4[1] !== last4[2] && last4[2] !== last4[3]) {
                    const streak = this.getStreak(results.slice(0, -1));
                    if (streak > 4) {
                        return {
                            prediction: last,
                            confidence: 0.8,
                            reason: 'Cầu 1-1 dài sắp gãy, dự đoán giữ nguyên'
                        };
                    }
                }
                break;
                
            case 'recovery':
                if (last4[0] === last4[1] && last4[1] !== last4[2] && last4[2] !== last4[3]) {
                    return {
                        prediction: last4[3] === 'Tài' ? 'Xỉu' : 'Tài',
                        confidence: 0.7,
                        reason: 'Cầu 1-1 đang phục hồi sau gãy'
                    };
                }
                break;
        }
        
        return null;
    }
    
    // Model 7-12: Chuyên cầu 2-2
    runSubModel22(results, model) {
        if (results.length < model.minLength) return null;
        
        const last = results[results.length - 1];
        const last4 = results.slice(-4);
        const last6 = results.slice(-6);
        const last8 = results.slice(-8);
        
        switch (model.logic) {
            case 'pure':
                if (last6.length === 6) {
                    if (last6[0] === last6[1] && last6[1] !== last6[2] &&
                        last6[2] === last6[3] && last6[3] !== last6[4] &&
                        last6[4] === last6[5]) {
                        return {
                            prediction: last6[4] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.9,
                            reason: 'Phát hiện cầu 2-2 chuẩn'
                        };
                    }
                }
                break;
                
            case 'offset':
                if (last6.length === 6) {
                    if (last6[0] === last6[1] && last6[1] !== last6[2] &&
                        last6[2] !== last6[3] && last6[3] === last6[4] &&
                        last6[4] !== last6[5]) {
                        return {
                            prediction: last6[4] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.8,
                            reason: 'Phát hiện cầu 2-2 lệch'
                        };
                    }
                }
                break;
                
            case 'variant':
                if (last8.length === 8) {
                    if (last8[0] === last8[1] && last8[1] !== last8[2] &&
                        last8[2] === last8[3] && last8[3] !== last8[4] &&
                        last8[4] === last8[5] && last8[5] !== last8[6] &&
                        last8[6] === last8[7]) {
                        return {
                            prediction: last8[6] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.75,
                            reason: 'Phát hiện cầu 2-2 biến tướng'
                        };
                    }
                }
                break;
                
            case 'hybrid':
                if (last6.length === 6) {
                    if (last6[0] === last6[1] && last6[1] !== last6[2] &&
                        last6[2] !== last6[3] && last6[3] !== last6[4] &&
                        last6[4] === last6[5]) {
                        return {
                            prediction: last6[4] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.7,
                            reason: 'Cầu 2-2 kết hợp 1-1'
                        };
                    }
                }
                break;
                
            case 'long':
                if (last8.length === 8) {
                    let score = 0;
                    for (let i = 0; i < 7; i+=2) {
                        if (last8[i] === last8[i+1]) score++;
                    }
                    if (score >= 3) {
                        return {
                            prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.7 + (score * 0.05),
                            reason: `Cầu 2-2 dài với ${score}/4 cặp đúng`
                        };
                    }
                }
                break;
                
            case 'break':
                if (last6.length === 6) {
                    if (last6[0] === last6[1] && last6[1] !== last6[2] &&
                        last6[2] === last6[3] && last6[3] !== last6[4] &&
                        last6[4] !== last6[5]) {
                        return {
                            prediction: last6[4],
                            confidence: 0.85,
                            reason: 'Phát hiện bẻ cầu 2-2'
                        };
                    }
                }
                break;
        }
        
        return null;
    }
    
    // Model 13-18: Chuyên cầu bệt
    runSubModelStreak(results, model) {
        if (results.length < model.minLength) return null;
        
        const last = results[results.length - 1];
        const other = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        let streak = 1;
        for (let i = results.length - 2; i >= 0; i--) {
            if (results[i] === last) streak++;
            else break;
        }
        
        switch (model.logic) {
            case 'short':
                if (streak >= 2 && streak <= 3) {
                    return {
                        prediction: last,
                        confidence: 0.7 + (streak * 0.05),
                        reason: `Bệt ngắn ${streak} phiên`
                    };
                }
                break;
                
            case 'medium':
                if (streak >= 4 && streak <= 5) {
                    return {
                        prediction: last,
                        confidence: 0.75 + ((streak - 4) * 0.05),
                        reason: `Bệt trung ${streak} phiên`
                    };
                }
                break;
                
            case 'long':
                if (streak >= 6) {
                    return {
                        prediction: last,
                        confidence: 0.8 + (Math.min(streak, 10) * 0.01),
                        reason: `Bệt dài ${streak} phiên`
                    };
                }
                break;
                
            case 'break':
                if (streak >= 4) {
                    return {
                        prediction: other,
                        confidence: 0.6 + (streak * 0.03),
                        reason: `Bệt ${streak} phiên, dự đoán sắp gãy`
                    };
                }
                break;
                
            case 'hybrid':
                if (streak >= 3) {
                    const prev = results[results.length - streak - 1];
                    if (prev && prev !== last) {
                        return {
                            prediction: last,
                            confidence: 0.7,
                            reason: `Bệt sau khi đảo từ ${prev}`
                        };
                    }
                }
                break;
                
            case 'super':
                if (streak >= 8) {
                    return {
                        prediction: last,
                        confidence: 0.9,
                        reason: `Siêu bệt ${streak} phiên`
                    };
                }
                break;
        }
        
        return null;
    }
    
    // Model 19-24: Chuyên cầu 3-3
    runSubModel33(results, model) {
        if (results.length < model.minLength) return null;
        
        const last = results[results.length - 1];
        const last9 = results.slice(-9);
        const last12 = results.slice(-12);
        
        switch (model.logic) {
            case 'pure':
                if (last9.length === 9) {
                    if (last9[0] === last9[1] && last9[1] === last9[2] &&
                        last9[3] === last9[4] && last9[4] === last9[5] &&
                        last9[6] === last9[7] && last9[7] === last9[8] &&
                        last9[0] !== last9[3] && last9[3] !== last9[6]) {
                        return {
                            prediction: last9[6] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.9,
                            reason: 'Phát hiện cầu 3-3 chuẩn'
                        };
                    }
                }
                break;
                
            case 'variant':
                if (last12.length === 12) {
                    let score = 0;
                    for (let i = 0; i < 12; i+=3) {
                        if (i+2 < 12 && last12[i] === last12[i+1] && last12[i+1] === last12[i+2]) {
                            score++;
                        }
                    }
                    if (score >= 3) {
                        return {
                            prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.7 + (score * 0.05),
                            reason: `Cầu 3-3 biến thể với ${score}/4 bộ ba`
                        };
                    }
                }
                break;
                
            case 'short':
                if (results.length >= 6) {
                    const last6 = results.slice(-6);
                    if (last6[0] === last6[1] && last6[1] === last6[2] &&
                        last6[3] === last6[4] && last6[4] === last6[5]) {
                        return {
                            prediction: last6[3] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.7,
                            reason: 'Cầu 3-3 ngắn (6 phiên)'
                        };
                    }
                }
                break;
                
            case 'hybrid':
                if (last9.length === 9) {
                    if (last9[0] === last9[1] && last9[1] === last9[2] &&
                        last9[3] !== last9[4] && last9[5] === last9[6] && last9[6] === last9[7]) {
                        return {
                            prediction: last9[6] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.75,
                            reason: 'Cầu 3-3 kết hợp'
                        };
                    }
                }
                break;
                
            case 'break':
                if (last9.length === 9) {
                    if (last9[0] === last9[1] && last9[1] === last9[2] &&
                        last9[3] === last9[4] && last9[4] === last9[5] &&
                        last9[6] !== last9[7]) {
                        return {
                            prediction: last9[6],
                            confidence: 0.8,
                            reason: 'Phát hiện bẻ cầu 3-3'
                        };
                    }
                }
                break;
                
            case 'long':
                if (results.length >= 15) {
                    const last15 = results.slice(-15);
                    let pattern = [];
                    for (let i = 0; i < 15; i+=3) {
                        if (i+2 < 15 && last15[i] === last15[i+1] && last15[i+1] === last15[i+2]) {
                            pattern.push(last15[i]);
                        }
                    }
                    if (pattern.length >= 4 && pattern[0] !== pattern[1] && pattern[1] !== pattern[2]) {
                        return {
                            prediction: pattern[pattern.length-1] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.8,
                            reason: 'Cầu 3-3 dài hạn'
                        };
                    }
                }
                break;
        }
        
        return null;
    }
    
    // Model 25-30: Chuyên cầu 2-1-2 và 1-2-1
    runSubModel212(results, model) {
        if (results.length < model.minLength) return null;
        
        const last = results[results.length - 1];
        const last5 = results.slice(-5);
        const last7 = results.slice(-7);
        
        switch (model.logic) {
            case 'pure':
                if (last5.length === 5) {
                    if (last5[0] === last5[1] && last5[1] !== last5[2] &&
                        last5[2] !== last5[3] && last5[3] === last5[4] &&
                        last5[0] === last5[3]) {
                        return {
                            prediction: last5[4] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.9,
                            reason: 'Phát hiện cầu 2-1-2 chuẩn'
                        };
                    }
                }
                break;
                
            case 'variant':
                if (last7.length === 7) {
                    if (last7[0] === last7[1] && last7[1] !== last7[2] &&
                        last7[3] === last7[4] && last7[4] !== last7[5] &&
                        last7[0] === last7[3]) {
                        return {
                            prediction: last7[5] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.8,
                            reason: 'Phát hiện cầu 2-1-2 biến thể'
                        };
                    }
                }
                break;
                
            case 'long':
                if (results.length >= 10) {
                    const last10 = results.slice(-10);
                    let count = 0;
                    for (let i = 0; i < 5; i+=2) {
                        if (i+4 < 10 && last10[i] === last10[i+1] && last10[i+1] !== last10[i+2] &&
                            last10[i+3] === last10[i+4]) {
                            count++;
                        }
                    }
                    if (count >= 2) {
                        return {
                            prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.75,
                            reason: 'Cầu 2-1-2 dài hạn'
                        };
                    }
                }
                break;
        }
        
        return null;
    }
    
    runSubModel121(results, model) {
        if (results.length < model.minLength) return null;
        
        const last = results[results.length - 1];
        const last5 = results.slice(-5);
        const last7 = results.slice(-7);
        
        switch (model.logic) {
            case 'pure':
                if (last5.length === 5) {
                    if (last5[0] !== last5[1] && last5[1] === last5[2] &&
                        last5[2] !== last5[3] && last5[3] === last5[4] &&
                        last5[0] === last5[3]) {
                        return {
                            prediction: last5[4] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.9,
                            reason: 'Phát hiện cầu 1-2-1 chuẩn'
                        };
                    }
                }
                break;
                
            case 'variant':
                if (last7.length === 7) {
                    if (last7[0] !== last7[1] && last7[1] === last7[2] &&
                        last7[3] !== last7[4] && last7[4] === last7[5] &&
                        last7[0] === last7[3]) {
                        return {
                            prediction: last7[5] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.8,
                            reason: 'Phát hiện cầu 1-2-1 biến thể'
                        };
                    }
                }
                break;
                
            case 'long':
                if (results.length >= 10) {
                    const last10 = results.slice(-10);
                    let count = 0;
                    for (let i = 0; i < 5; i+=2) {
                        if (i+4 < 10 && last10[i] !== last10[i+1] && last10[i+1] === last10[i+2] &&
                            last10[i+3] === last10[i+4]) {
                            count++;
                        }
                    }
                    if (count >= 2) {
                        return {
                            prediction: last === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.75,
                            reason: 'Cầu 1-2-1 dài hạn'
                        };
                    }
                }
                break;
        }
        
        return null;
    }
    
    // Model 31-36: Chuyên bẻ cầu và chuyển tiếp
    runSubModelBreak(results, model) {
        if (results.length < model.minLength) return null;
        
        const last = results[results.length - 1];
        const last4 = results.slice(-4);
        const last5 = results.slice(-5);
        const last6 = results.slice(-6);
        
        switch (model.logic) {
            case 'break11':
                if (last4.length === 4) {
                    if (last4[0] !== last4[1] && last4[1] !== last4[2] && last4[2] === last4[3]) {
                        return {
                            prediction: last4[3],
                            confidence: 0.85,
                            reason: 'Phát hiện bẻ cầu 1-1'
                        };
                    }
                }
                break;
                
            case 'break22':
                if (last5.length === 5) {
                    if (last5[0] === last5[1] && last5[1] !== last5[2] &&
                        last5[2] === last5[3] && last5[3] !== last5[4] &&
                        last5[0] === last5[4]) {
                        return {
                            prediction: last5[4],
                            confidence: 0.85,
                            reason: 'Phát hiện bẻ cầu 2-2'
                        };
                    }
                }
                break;
                
            case 'breakStreak':
                const streak = this.getStreak(results.slice(0, -1));
                if (streak >= 3 && last !== results[results.length - 2]) {
                    return {
                        prediction: last,
                        confidence: 0.8,
                        reason: `Phát hiện bẻ cầu bệt sau ${streak} phiên`
                    };
                }
                break;
                
            case '11to22':
                if (last6.length === 6) {
                    if (last6[0] !== last6[1] && last6[1] !== last6[2] &&
                        last6[2] === last6[3] && last6[3] !== last6[4] &&
                        last6[4] === last6[5]) {
                        return {
                            prediction: last6[4] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.75,
                            reason: 'Chuyển từ cầu 1-1 sang 2-2'
                        };
                    }
                }
                break;
                
            case '22to11':
                if (last6.length === 6) {
                    if (last6[0] === last6[1] && last6[1] !== last6[2] &&
                        last6[2] !== last6[3] && last6[3] !== last6[4] &&
                        last6[4] !== last6[5]) {
                        return {
                            prediction: last6[4] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.75,
                            reason: 'Chuyển từ cầu 2-2 sang 1-1'
                        };
                    }
                }
                break;
                
            case 'streakTo11':
                if (last5.length === 5) {
                    if (last5[0] === last5[1] && last5[1] === last5[2] &&
                        last5[2] !== last5[3] && last5[3] !== last5[4]) {
                        return {
                            prediction: last5[4] === 'Tài' ? 'Xỉu' : 'Tài',
                            confidence: 0.7,
                            reason: 'Chuyển từ bệt sang cầu 1-1'
                        };
                    }
                }
                break;
        }
        
        return null;
    }
    
    // Model 37-42: Chuyên phân tích tổng hợp
    runSubModelAdvanced(results, model) {
        if (results.length < model.minLength) return null;
        
        const last = results[results.length - 1];
        const other = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        switch (model.logic) {
            case 'frequency':
                const freq = this.analyzeFrequency(results);
                if (freq.dominant && freq.ratio > 0.6) {
                    return {
                        prediction: freq.dominant,
                        confidence: 0.6 + (freq.ratio * 0.2),
                        reason: `Tần suất ${freq.dominant} chiếm ${(freq.ratio*100).toFixed(0)}%`
                    };
                }
                break;
                
            case 'cycle':
                const cycle = this.detectCycle(results);
                if (cycle.found) {
                    return {
                        prediction: cycle.next,
                        confidence: 0.7,
                        reason: `Phát hiện chu kỳ ${cycle.length} phiên`
                    };
                }
                break;
                
            case 'symmetry':
                const symmetry = this.checkSymmetry(results);
                if (symmetry.found) {
                    return {
                        prediction: symmetry.prediction,
                        confidence: 0.75,
                        reason: 'Phát hiện cầu đối xứng'
                    };
                }
                break;
                
            case 'fibonacci':
                const fib = this.checkFibonacci(results);
                if (fib.found) {
                    return {
                        prediction: fib.prediction,
                        confidence: 0.7,
                        reason: 'Phát hiện cầu Fibonacci'
                    };
                }
                break;
                
            case 'longTrend':
                const trend = this.getLongTrend(results);
                if (trend.strength > 0.7) {
                    return {
                        prediction: trend.direction,
                        confidence: 0.7 + (trend.strength * 0.1),
                        reason: `Xu hướng dài ${trend.direction} với độ mạnh ${(trend.strength*100).toFixed(0)}%`
                    };
                }
                break;
                
            case 'super':
                const superAnalysis = this.superAnalysis(results);
                if (superAnalysis.confidence > 0.8) {
                    return superAnalysis;
                }
                break;
        }
        
        return null;
    }
    
    // Helper functions
    isPerfectAlternating(results, length) {
        const last = results.slice(-length);
        for (let i = 0; i < last.length - 1; i++) {
            if (last[i] === last[i+1]) return false;
        }
        return true;
    }
    
    isAlternatingWithTolerance(results, tolerance) {
        const last = results.slice(-6);
        let errors = 0;
        for (let i = 0; i < last.length - 1; i++) {
            if (last[i] === last[i+1]) errors++;
        }
        return errors <= tolerance;
    }
    
    countAlternating(results) {
        let count = 0;
        for (let i = 0; i < results.length - 1; i++) {
            if (results[i] !== results[i+1]) count++;
        }
        return count;
    }
    
    getStreak(results) {
        if (results.length === 0) return 0;
        const last = results[results.length - 1];
        let streak = 1;
        for (let i = results.length - 2; i >= 0; i--) {
            if (results[i] === last) streak++;
            else break;
        }
        return streak;
    }
    
    analyzeFrequency(results) {
        const recent = results.slice(-20);
        const taiCount = recent.filter(r => r === 'Tài').length;
        const xiuCount = recent.length - taiCount;
        const ratio = Math.max(taiCount, xiuCount) / recent.length;
        const dominant = taiCount > xiuCount ? 'Tài' : 'Xỉu';
        return { dominant, ratio };
    }
    
    detectCycle(results) {
        for (let cycleLen of [2, 3, 4]) {
            if (results.length < cycleLen * 2) continue;
            const lastCycle = results.slice(-cycleLen);
            const prevCycle = results.slice(-cycleLen*2, -cycleLen);
            if (JSON.stringify(lastCycle) === JSON.stringify(prevCycle)) {
                return {
                    found: true,
                    length: cycleLen,
                    next: lastCycle[0]
                };
            }
        }
        return { found: false };
    }
    
    checkSymmetry(results) {
        if (results.length < 6) return { found: false };
        const last3 = results.slice(-3);
        const prev3 = results.slice(-6, -3);
        if (last3[0] === prev3[2] && last3[1] === prev3[1] && last3[2] === prev3[0]) {
            return {
                found: true,
                prediction: last3[1]
            };
        }
        return { found: false };
    }
    
    checkFibonacci(results) {
        if (results.length < 5) return { found: false };
        const fibs = [1, 2, 3, 5];
        for (let fib of fibs) {
            if (results.length >= fib * 2) {
                const lastFib = results.slice(-fib);
                const prevFib = results.slice(-fib*2, -fib);
                if (JSON.stringify(lastFib) === JSON.stringify(prevFib)) {
                    return {
                        found: true,
                        prediction: lastFib[0]
                    };
                }
            }
        }
        return { found: false };
    }
    
    getLongTrend(results) {
        if (results.length < 10) return { strength: 0, direction: null };
        const first = results.slice(0, 5);
        const last = results.slice(-5);
        const firstTai = first.filter(r => r === 'Tài').length;
        const lastTai = last.filter(r => r === 'Tài').length;
        
        if (lastTai > firstTai + 2) {
            return { strength: 0.8, direction: 'Tài' };
        } else if (lastTai < firstTai - 2) {
            return { strength: 0.8, direction: 'Xỉu' };
        }
        return { strength: 0.5, direction: lastTai > 2 ? 'Tài' : 'Xỉu' };
    }
    
    superAnalysis(results) {
        const freq = this.analyzeFrequency(results);
        const trend = this.getLongTrend(results);
        const cycle = this.detectCycle(results);
        
        let score = 0;
        let predictions = [];
        
        if (freq.ratio > 0.6) {
            predictions.push({ pred: freq.dominant, weight: freq.ratio });
            score++;
        }
        
        if (trend.strength > 0.7) {
            predictions.push({ pred: trend.direction, weight: trend.strength });
            score++;
        }
        
        if (cycle.found) {
            predictions.push({ pred: cycle.next, weight: 0.7 });
            score++;
        }
        
        if (score >= 2) {
            const taiWeight = predictions.filter(p => p.pred === 'Tài')
                .reduce((sum, p) => sum + p.weight, 0);
            const xiuWeight = predictions.filter(p => p.pred === 'Xỉu')
                .reduce((sum, p) => sum + p.weight, 0);
            
            if (taiWeight > xiuWeight * 1.5) {
                return {
                    prediction: 'Tài',
                    confidence: 0.85,
                    reason: 'Siêu phân tích đồng thuận Tài'
                };
            } else if (xiuWeight > taiWeight * 1.5) {
                return {
                    prediction: 'Xỉu',
                    confidence: 0.85,
                    reason: 'Siêu phân tích đồng thuận Xỉu'
                };
            }
        }
        
        return { confidence: 0 };
    }
    
    // Run sub model
    runSubModel(index, history) {
        if (history.length < 3) return null;
        
        const results = this.getResultArray(history);
        const model = this.subModels[`sub_model_${index}`];
        
        if (!model) return null;
        
        let result = null;
        const type = model.type;
        
        switch (type) {
            case '1-1':
                result = this.runSubModel11(results, model);
                break;
            case '2-2':
                result = this.runSubModel22(results, model);
                break;
            case 'bệt':
                result = this.runSubModelStreak(results, model);
                break;
            case '3-3':
                result = this.runSubModel33(results, model);
                break;
            case '2-1-2':
                result = this.runSubModel212(results, model);
                break;
            case '1-2-1':
                result = this.runSubModel121(results, model);
                break;
            case 'break':
            case 'transition':
                result = this.runSubModelBreak(results, model);
                break;
            default:
                result = this.runSubModelAdvanced(results, model);
        }
        
        if (result) {
            result.model_name = model.name;
            return result;
        }
        
        return null;
    }
    
    // Run mini model
    runMiniModel(index, history) {
        if (history.length < 2) return null;
        
        const results = this.getResultArray(history);
        const miniModel = this.miniModels[`mini_model_${index}`];
        
        let prediction, confidence, reason;
        
        switch (miniModel.specialty) {
            case 'phat_hien_cau_dep':
                const pattern = this.analyzeBasicPatterns(history);
                prediction = pattern.prediction;
                confidence = pattern.confidence * 0.9;
                reason = pattern.reason;
                break;
                
            case 'du_doan_bien_dong':
                const dice = this.analyzeDiceVolatility(history);
                prediction = dice.prediction;
                confidence = dice.confidence * 0.8;
                reason = dice.reason;
                break;
                
            case 'nhan_dien_xu_huong_cuc_bo':
                const short = this.analyzeShortTerm(history);
                prediction = short.prediction;
                confidence = short.confidence * 0.85;
                reason = short.reason;
                break;
                
            case 'tinh_toan_xac_suat_cao':
                const taiCount = results.filter(r => r === 'Tài').length;
                const xiuCount = results.length - taiCount;
                if (taiCount > xiuCount * 1.5) {
                    prediction = 'Xỉu';
                    confidence = 0.7;
                    reason = 'Xác suất Tài cao, dự đoán Xỉu để cân bằng';
                } else if (xiuCount > taiCount * 1.5) {
                    prediction = 'Tài';
                    confidence = 0.7;
                    reason = 'Xác suất Xỉu cao, dự đoán Tài để cân bằng';
                } else {
                    prediction = results[results.length - 1];
                    confidence = 0.5;
                    reason = 'Xác suất cân bằng';
                }
                break;
                
            case 'phan_tich_so_sanh':
                const currentPattern = results.slice(-5).join('');
                let matchFound = false;
                for (let [type, patterns] of Object.entries(this.patternLibrary)) {
                    if (patterns.includes(currentPattern)) {
                        matchFound = true;
                        prediction = results[results.length - 1] === 'Tài' ? 'Xỉu' : 'Tài';
                        confidence = 0.75;
                        reason = `Khớp mẫu ${type} trong thư viện`;
                        break;
                    }
                }
                if (!matchFound) {
                    prediction = results[results.length - 1];
                    confidence = 0.4;
                    reason = 'Không tìm thấy mẫu tương tự';
                }
                break;
                
            default:
                const random = Math.random();
                if (random < 0.4) {
                    prediction = results[results.length - 1];
                    confidence = 0.5;
                } else if (random < 0.7) {
                    prediction = results[results.length - 1] === 'Tài' ? 'Xỉu' : 'Tài';
                    confidence = 0.5;
                } else {
                    const streak = this.getStreak(results);
                    if (streak >= 3) {
                        prediction = results[results.length - 1];
                        confidence = 0.6;
                    } else {
                        prediction = results[results.length - 1] === 'Tài' ? 'Xỉu' : 'Tài';
                        confidence = 0.5;
                    }
                }
                reason = `Mini model ${index} (${miniModel.specialty})`;
        }
        
        return {
            prediction,
            confidence: Math.min(confidence, 0.95),
            reason,
            model_name: `mini_${index}_${miniModel.specialty}`
        };
    }
    
    // ==================== CÁC MODEL CHÍNH (1-4, 11 cũ) ====================
    
    analyzeBasicPatterns(history) {
        if (history.length < 3) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu' };
        }
        
        const results = this.getResultArray(history);
        
        const patterns = {
            '1-1': this.checkAlternatingPattern(results),
            '1-2-1': this.checkPattern121(results),
            '2-1-2': this.checkPattern212(results),
            '3-1': this.checkPattern31(results),
            '1-3': this.checkPattern13(results),
            '2-2': this.checkPattern22(results),
            'cầu_bệt': this.checkStreakPattern(results),
            'cầu_đảo': this.checkReversalPattern(results)
        };
        
        const validPatterns = {};
        for (let [key, value] of Object.entries(patterns)) {
            if (value && value.confidence > 0) {
                validPatterns[key] = value;
            }
        }
        
        if (Object.keys(validPatterns).length === 0) {
            return {
                prediction: results[results.length - 1],
                confidence: 0.3,
                reason: 'Không phát hiện pattern rõ ràng'
            };
        }
        
        let bestPattern = null;
        let bestConfidence = 0;
        let bestKey = '';
        
        for (let [key, value] of Object.entries(validPatterns)) {
            if (value.confidence > bestConfidence) {
                bestConfidence = value.confidence;
                bestPattern = value;
                bestKey = key;
            }
        }
        
        return {
            prediction: bestPattern.prediction,
            confidence: bestPattern.confidence,
            pattern_type: bestKey,
            reason: `Phát hiện cầu ${bestKey} với độ tin cậy ${(bestPattern.confidence * 100).toFixed(0)}%`
        };
    }
    
    checkAlternatingPattern(results) {
        if (results.length < 2) return { prediction: null, confidence: 0 };
        
        const last = results[results.length - 1];
        const pred = last === 'Tài' ? 'Xỉu' : 'Tài';
        
        let confidence = 0.5;
        for (let i = results.length - 2; i >= Math.max(results.length - 6, 0); i -= 2) {
            if (results[i] === last) {
                confidence += 0.1;
            } else break;
        }
        
        return { prediction: pred, confidence: Math.min(confidence, 0.95) };
    }
    
    checkPattern121(results) {
        if (results.length < 3) return { prediction: null, confidence: 0 };
        if (results[results.length - 3] === results[results.length - 1] && 
            results[results.length - 2] !== results[results.length - 1]) {
            return { prediction: results[results.length - 1], confidence: 0.7 };
        }
        return { prediction: results[results.length - 1], confidence: 0.3 };
    }
    
    checkPattern212(results) {
        if (results.length < 3) return { prediction: null, confidence: 0 };
        if (results[results.length - 3] !== results[results.length - 1] && 
            results[results.length - 2] === results[results.length - 1]) {
            return { prediction: results[results.length - 2], confidence: 0.7 };
        }
        return { prediction: results[results.length - 1], confidence: 0.3 };
    }
    
    checkPattern31(results) {
        if (results.length < 4) return { prediction: null, confidence: 0 };
        if (results[results.length - 4] === results[results.length - 3] && 
            results[results.length - 3] === results[results.length - 2] && 
            results[results.length - 2] !== results[results.length - 1]) {
            return { prediction: results[results.length - 1], confidence: 0.8 };
        }
        return { prediction: results[results.length - 1], confidence: 0.2 };
    }
    
    checkPattern13(results) {
        if (results.length < 4) return { prediction: null, confidence: 0 };
        if (results[results.length - 4] !== results[results.length - 3] && 
            results[results.length - 3] === results[results.length - 2] && 
            results[results.length - 2] === results[results.length - 1]) {
            return { prediction: results[results.length - 1], confidence: 0.8 };
        }
        return { prediction: results[results.length - 1], confidence: 0.2 };
    }
    
    checkPattern22(results) {
        if (results.length < 4) return { prediction: null, confidence: 0 };
        if (results[results.length - 4] === results[results.length - 3] && 
            results[results.length - 2] === results[results.length - 1] && 
            results[results.length - 3] !== results[results.length - 2]) {
            return { prediction: results[results.length - 1], confidence: 0.75 };
        }
        return { prediction: results[results.length - 1], confidence: 0.25 };
    }
    
    checkStreakPattern(results) {
        let streak = 1;
        for (let i = results.length - 2; i >= 0; i--) {
            if (results[i] === results[results.length - 1]) streak++;
            else break;
        }
        
        if (streak >= 3) {
            let confidence = 0.6 + (streak * 0.05);
            return { prediction: results[results.length - 1], confidence: Math.min(confidence, 0.9) };
        }
        const other = results[results.length - 1] === 'Tài' ? 'Xỉu' : 'Tài';
        if (streak >= 6) return { prediction: other, confidence: 0.65 };
        return { prediction: results[results.length - 1], confidence: 0.4 };
    }
    
    checkReversalPattern(results) {
        if (results.length < 3) return { prediction: null, confidence: 0 };
        if (results[results.length - 2] !== results[results.length - 1]) {
            return { prediction: results[results.length - 1], confidence: 0.5 };
        }
        const other = results[results.length - 1] === 'Tài' ? 'Xỉu' : 'Tài';
        return { prediction: other, confidence: 0.4 };
    }
    
    analyzeTrend(history) {
        if (history.length < 5) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu' };
        }
        
        const results = this.getResultArray(history);
        const shortTerm = results.slice(-3);
        const shortCounts = this.countResults(shortTerm);
        const shortTrend = this.getMostCommon(shortCounts);
        const longTerm = results.slice(-10);
        const longCounts = this.countResults(longTerm);
        const longTrend = this.getMostCommon(longCounts);
        const momentum = this.calculateMomentum(results);
        
        if (shortTrend.count >= 2 && longTrend.count >= 6) {
            return {
                prediction: shortTrend.value,
                confidence: Math.min(0.7 + momentum * 0.1, 0.95),
                momentum: momentum,
                reason: `Xu hướng ngắn và dài đều nghiêng về ${shortTrend.value}`
            };
        } else if (shortTrend.count >= 2) {
            return {
                prediction: shortTrend.value,
                confidence: Math.min(0.6 + momentum * 0.1, 0.95),
                momentum: momentum,
                reason: `Xu hướng ngắn hạn nghiêng về ${shortTrend.value}`
            };
        } else if (longTrend.count >= 6) {
            return {
                prediction: longTrend.value,
                confidence: Math.min(0.6 + momentum * 0.1, 0.95),
                momentum: momentum,
                reason: `Xu hướng dài hạn nghiêng về ${longTrend.value}`
            };
        }
        const other = results[results.length - 1] === 'Tài' ? 'Xỉu' : 'Tài';
        return {
            prediction: other,
            confidence: 0.5,
            momentum: momentum,
            reason: "Không có trend rõ ràng, dự đoán đảo chiều"
        };
    }
    
    countResults(results) {
        const counts = { 'Tài': 0, 'Xỉu': 0 };
        results.forEach(r => counts[r]++);
        return counts;
    }
    
    getMostCommon(counts) {
        if (counts['Tài'] >= counts['Xỉu']) {
            return { value: 'Tài', count: counts['Tài'] };
        }
        return { value: 'Xỉu', count: counts['Xỉu'] };
    }
    
    calculateMomentum(results) {
        if (results.length < 5) return 0;
        const recent = results.slice(-5);
        const taiCount = recent.filter(r => r === 'Tài').length;
        if (taiCount === 5 || taiCount === 0) return 0.3;
        if (taiCount >= 3 || taiCount <= 2) return 0.15;
        return 0;
    }
    
    analyzeImbalance(history) {
        if (history.length < 12) {
            return { prediction: null, confidence: 0, reason: 'Không đủ 12 phiên' };
        }
        
        const results = this.getResultArray(history.slice(-12));
        const countTai = results.filter(r => r === 'Tài').length;
        const countXiu = results.length - countTai;
        const imbalanceRatio = Math.abs(countTai - countXiu) / 12;
        
        if (imbalanceRatio > 0.4) {
            if (countTai > countXiu) {
                return {
                    prediction: 'Xỉu',
                    confidence: Math.min(0.7 + imbalanceRatio * 0.2, 0.95),
                    tai_count: countTai,
                    xiu_count: countXiu,
                    reason: `Chênh lệch lớn (${countTai}T - ${countXiu}X), dự đoán Xỉu để cân bằng`
                };
            }
            return {
                prediction: 'Tài',
                confidence: Math.min(0.7 + imbalanceRatio * 0.2, 0.95),
                tai_count: countTai,
                xiu_count: countXiu,
                reason: `Chênh lệch lớn (${countTai}T - ${countXiu}X), dự đoán Tài để cân bằng`
            };
        }
        return {
            prediction: results[results.length - 1],
            confidence: 0.5,
            tai_count: countTai,
            xiu_count: countXiu,
            reason: `Chênh lệch ${countTai}T - ${countXiu}X trong 12 phiên, tiếp tục xu hướng`
        };
    }
    
    analyzeShortTerm(history) {
        if (history.length < 3) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu' };
        }
        
        const results = this.getResultArray(history);
        const last3 = results.slice(-3);
        const patterns = [];
        
        if (last3[0] === last3[1] && last3[1] === last3[2]) {
            patterns.push({ type: 'bệt', prediction: last3[0], confidence: 0.75 });
        }
        if (last3[0] === last3[1] && last3[1] !== last3[2]) {
            patterns.push({ type: '2-1', prediction: last3[2], confidence: 0.7 });
        }
        if (last3[0] !== last3[1] && last3[1] === last3[2]) {
            const other = last3[2] === 'Tài' ? 'Xỉu' : 'Tài';
            patterns.push({ type: '1-2', prediction: other, confidence: 0.65 });
        }
        if (results.length >= 4) {
            const last4 = results.slice(-4);
            if (last4[0] !== last4[1] && last4[1] !== last4[2] && last4[2] !== last4[3]) {
                const other = last4[3] === 'Tài' ? 'Xỉu' : 'Tài';
                patterns.push({ type: 'xen_kẽ', prediction: other, confidence: 0.8 });
            }
        }
        
        if (patterns.length > 0) {
            const bestPattern = patterns.reduce((best, current) => 
                current.confidence > best.confidence ? current : best
            );
            return {
                prediction: bestPattern.prediction,
                confidence: bestPattern.confidence,
                pattern: bestPattern.type,
                reason: `Phát hiện pattern ${bestPattern.type} trong ngắn hạn`
            };
        }
        return {
            prediction: results[results.length - 1],
            confidence: 0.4,
            pattern: 'không_rõ',
            reason: "Không phát hiện pattern ngắn hạn rõ ràng"
        };
    }
    
    analyzeDiceVolatility(history) {
        if (history.length < 5) {
            return { prediction: null, confidence: 0, reason: 'Không đủ dữ liệu' };
        }
        
        const faceSequences = [];
        history.forEach(h => {
            if (h.Xuc_xac_1) faceSequences.push(h.Xuc_xac_1);
            if (h.Xuc_xac_2) faceSequences.push(h.Xuc_xac_2);
            if (h.Xuc_xac_3) faceSequences.push(h.Xuc_xac_3);
        });
        
        if (faceSequences.length === 0) {
            return { prediction: null, confidence: 0, reason: 'Không có dữ liệu mặt xúc xắc' };
        }
        
        const faceFreq = {};
        for (let i = 1; i <= 6; i++) faceFreq[i] = 0;
        faceSequences.forEach(f => faceFreq[f]++);
        
        const recentFaces = [];
        const recentHistory = history.slice(-5);
        recentHistory.forEach(h => {
            if (h.Xuc_xac_1) recentFaces.push(h.Xuc_xac_1);
            if (h.Xuc_xac_2) recentFaces.push(h.Xuc_xac_2);
            if (h.Xuc_xac_3) recentFaces.push(h.Xuc_xac_3);
        });
        
        const recentFreq = {};
        for (let i = 1; i <= 6; i++) recentFreq[i] = 0;
        recentFaces.forEach(f => recentFreq[f]++);
        
        const predictions = [];
        for (let face = 1; face <= 6; face++) {
            if (recentFreq[face] < 2) {
                predictions.push({ face, prob: 0.3 + (2 - recentFreq[face]) * 0.1 });
            }
        }
        
        if (predictions.length > 0) {
            predictions.sort((a, b) => b.prob - a.prob);
            const topFaces = predictions.slice(0, 3);
            
            const predictedScores = [];
            for (let i = 0; i < topFaces.length; i++) {
                for (let j = i; j < topFaces.length; j++) {
                    for (let k = j; k < topFaces.length; k++) {
                        predictedScores.push(topFaces[i].face + topFaces[j].face + topFaces[k].face);
                    }
                }
            }
            
            if (predictedScores.length > 0) {
                const avgPredicted = predictedScores.reduce((a, b) => a + b, 0) / predictedScores.length;
                const predType = avgPredicted >= 11 ? 'Tài' : 'Xỉu';
                return {
                    prediction: predType,
                    confidence: 0.65,
                    predicted_faces: topFaces.map(f => f.face),
                    reason: `Dựa trên biến động xúc xắc, các mặt ${topFaces.map(f => f.face).join(',')} có khả năng xuất hiện cao`
                };
            }
        }
        
        return {
            prediction: history[history.length - 1].Ket_qua || (history[history.length - 1].score >= 11 ? 'Tài' : 'Xỉu'),
            confidence: 0.4,
            reason: "Không phát hiện biến động đặc biệt"
        };
    }
    
    // NÂNG CẤP: Ensemble với Neural Network
    ensembleModels(history) {
        const modelResults = {};
        
        // Chạy các model chính (1-21)
        const mainResults = this.getAllMainModelResults(history);
        Object.assign(modelResults, mainResults);
        
        // Chạy sub models (1-42)
        for (let i = 1; i <= 42; i++) {
            const subResult = this.runSubModel(i, history);
            if (subResult && subResult.prediction) {
                modelResults[`sub_model_${i}`] = subResult;
            }
        }
        
        // Chạy mini models (1-21)
        for (let i = 1; i <= 21; i++) {
            const miniResult = this.runMiniModel(i, history);
            if (miniResult && miniResult.prediction) {
                modelResults[`mini_model_${i}`] = miniResult;
            }
        }
        
        // Tính weighted vote từ tất cả models (cách cũ)
        let taiWeight = 0;
        let xiuWeight = 0;
        let totalWeight = 0;
        let details = [];
        
        for (let [modelName, result] of Object.entries(modelResults)) {
            if (result && result.prediction && result.confidence > 0.3) {
                let weight = 1.0;
                if (modelName.startsWith('sub')) {
                    weight = this.subModelWeights[modelName] || 1.0;
                } else if (modelName.startsWith('mini')) {
                    weight = this.miniModelWeights[modelName] || 1.0;
                } else {
                    weight = this.modelWeights[modelName] || 1.0;
                }
                
                const weightedConfidence = weight * result.confidence;
                
                if (result.prediction === 'Tài') {
                    taiWeight += weightedConfidence;
                } else if (result.prediction === 'Xỉu') {
                    xiuWeight += weightedConfidence;
                }
                
                totalWeight += weightedConfidence;
                details.push({
                    model: result.model_name || modelName,
                    prediction: result.prediction,
                    confidence: result.confidence,
                    weight: weight,
                    reason: result.reason
                });
            }
        }
        
        details.sort((a, b) => b.confidence - a.confidence);
        
        // NÂNG CẤP: Neural Network prediction
        const neuralInput = this.getNeuralInput(history);
        const neuralOutput = neuralForward(neuralInput);
        
        // Kết hợp neural prediction với ensemble
        const neuralWeight = Math.min(0.3, stats.neuralTotal > 20 ? 
            (stats.neuralAccuracy * 0.3) : 0.1);
        
        if (neuralOutput.prediction === 'Tài') {
            taiWeight += neuralOutput.confidence * neuralWeight * 2;
        } else {
            xiuWeight += neuralOutput.confidence * neuralWeight * 2;
        }
        totalWeight += neuralOutput.confidence * neuralWeight * 2;
        
        details.push({
            model: 'Neural Network (84→42→2)',
            prediction: neuralOutput.prediction,
            confidence: neuralOutput.confidence,
            weight: neuralWeight,
            reason: `Neural network dự đoán (độ chính xác: ${(stats.neuralAccuracy*100).toFixed(1)}%)`
        });
        
        // Quyết định cuối cùng
        let finalPrediction, finalConfidence, finalReason, finalPattern, finalType;
        
        if (totalWeight > 0) {
            const taiRatio = taiWeight / totalWeight;
            const xiuRatio = xiuWeight / totalWeight;
            
            if (taiRatio > 0.55) {
                finalPrediction = 'Tài';
                finalConfidence = taiRatio;
                finalReason = `${details.length} models đồng thuận Tài (${(taiRatio*100).toFixed(1)}%)`;
            } else if (xiuRatio > 0.55) {
                finalPrediction = 'Xỉu';
                finalConfidence = xiuRatio;
                finalReason = `${details.length} models đồng thuận Xỉu (${(xiuRatio*100).toFixed(1)}%)`;
            } else {
                const bestModel = details[0];
                if (bestModel) {
                    finalPrediction = bestModel.prediction;
                    finalConfidence = 0.5 + bestModel.confidence * 0.2;
                    finalReason = `Tỉ lệ cân bằng, dùng model ${bestModel.model}: ${bestModel.reason}`;
                } else {
                    finalPrediction = history.length > 0 ? 
                        (history[history.length - 1].Ket_qua || 
                         (history[history.length - 1].score >= 11 ? 'Tài' : 'Xỉu')) : 'Tài';
                    finalConfidence = 0.5;
                    finalReason = "Không có model nào đủ tin cậy";
                }
            }
        } else {
            finalPrediction = history.length > 0 ? 
                (history[history.length - 1].Ket_qua || 
                 (history[history.length - 1].score >= 11 ? 'Tài' : 'Xỉu')) : 'Tài';
            finalConfidence = 0.5;
            finalReason = "Không đủ dữ liệu model";
        }
        
        if (details.length > 0) {
            finalType = details[0].model;
            finalPattern = history.length > 0 ? 
                this.getResultArray(history.slice(-5)).join('') : '';
        } else {
            finalType = 'Không xác định';
            finalPattern = '';
        }
        
        return {
            prediction: finalPrediction,
            confidence: finalConfidence,
            reason: finalReason,
            pattern_type: finalType,
            pattern: finalPattern,
            details: details.slice(0, 5),
            neuralOutput: neuralOutput
        };
    }
    
    // NÂNG CẤP: Cập nhật trọng số model + huấn luyện neural network
    updateModelWeights(actual, predicted, confidence, historyForTraining) {
        const correct = (actual === predicted) ? 1 : 0;
        
        // Update main models
        for (let modelName in this.modelWeights) {
            if (correct) {
                this.modelWeights[modelName] = Math.min(this.modelWeights[modelName] * 1.01, 2.0);
            } else {
                this.modelWeights[modelName] = Math.max(this.modelWeights[modelName] * 0.99, 0.5);
            }
        }
        
        // Update sub models
        for (let modelName in this.subModelWeights) {
            if (correct) {
                this.subModelWeights[modelName] = Math.min(this.subModelWeights[modelName] * 1.005, 1.5);
            } else {
                this.subModelWeights[modelName] = Math.max(this.subModelWeights[modelName] * 0.995, 0.7);
            }
        }
        
        // Update mini models
        for (let modelName in this.miniModelWeights) {
            if (correct) {
                this.miniModelWeights[modelName] = Math.min(this.miniModelWeights[modelName] * 1.003, 1.3);
            } else {
                this.miniModelWeights[modelName] = Math.max(this.miniModelWeights[modelName] * 0.997, 0.8);
            }
        }
        
        // NÂNG CẤP: Huấn luyện neural network
        if (historyForTraining && historyForTraining.length > 5) {
            const neuralInput = this.getNeuralInput(historyForTraining);
            const neuralOutput = neuralForward(neuralInput);
            const target = actual === 'Tài' ? 0 : 1;
            
            neuralBackward(neuralInput, target, neuralOutput);
            
            // Cập nhật thống kê neural
            stats.neuralTotal++;
            if (neuralOutput.prediction === actual) {
                stats.neuralCorrect++;
            }
            stats.neuralAccuracy = stats.neuralTotal > 0 ? 
                stats.neuralCorrect / stats.neuralTotal : 0;
        }
        
        saveModelWeights();
    }
}

// Initialize analyzer
const analyzer = new TaiXiuAnalyzer();

// ==================== WEBSOCKET ====================
const WEBSOCKET_URL = "wss://websocket.azhkthg1.net/websocket?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhbW91bnQiOjAsInVzZXJuYW1lIjoiU0NfYXBpc3Vud2luMTIzIn0.hgrRbSV6vnBwJMg9ZFtbx3rRu9mX_hZMZ_m5gMNhkw0";
const WS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Origin": "https://play.sun.win"
};
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

    ws = new WebSocket(WEBSOCKET_URL, { headers: WS_HEADERS });

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

    ws.on('pong', () => {
        // console.log('[📶] Ping OK.');
    });

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
                    
                    // NÂNG CẤP: Tạo history cho training
                    const historyForTraining = resultHistory.slice(-50).map(h => ({
                        score: h.Tong,
                        Ket_qua: h.Ket_qua,
                        Xuc_xac_1: h.Xuc_xac_1,
                        Xuc_xac_2: h.Xuc_xac_2,
                        Xuc_xac_3: h.Xuc_xac_3
                    }));
                    
                    // Update model weights với training data
                    analyzer.updateModelWeights(result, lastPrediction.ket_qua, lastPrediction.do_tin_cay, historyForTraining);
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

                // Cập nhật pattern library
                if (resultHistory.length >= 6) {
                    const lastPattern = analyzer.getResultArray(
                        resultHistory.slice(-6).map(h => ({
                            Ket_qua: h.Ket_qua, score: h.Tong
                        }))
                    ).slice(0, 5).join('');
                    
                    const patternType = lastPrediction ? lastPrediction.loai_cau : 'không_xác_định';
                    if (analyzer.patternLibrary[patternType]) {
                        if (!analyzer.patternLibrary[patternType].includes(lastPattern)) {
                            analyzer.patternLibrary[patternType].push(lastPattern);
                            if (analyzer.patternLibrary[patternType].length > 100) {
                                analyzer.patternLibrary[patternType].shift();
                            }
                        }
                    }
                    analyzer.savePatternLibrary();
                }

                // Tạo mảng history cho analyzer
                const historyForAnalyzer = resultHistory.map(h => ({
                    score: h.Tong,
                    Ket_qua: h.Ket_qua,
                    Xuc_xac_1: h.Xuc_xac_1,
                    Xuc_xac_2: h.Xuc_xac_2,
                    Xuc_xac_3: h.Xuc_xac_3
                }));

                // Dự đoán cho phiên tiếp theo
                const ensembleResult = analyzer.ensembleModels(historyForAnalyzer);
                
                // Adjust for consecutive losses
                let finalPrediction = ensembleResult.prediction;
                let finalConfidence = ensembleResult.confidence;
                let finalType = ensembleResult.pattern_type;
                let finalPattern = ensembleResult.pattern;
                let finalReason = ensembleResult.reason;
                
                if (stats.consecutiveLosses >= 3) {
                    finalPrediction = finalPrediction === 'Tài' ? 'Xỉu' : 'Tài';
                    finalConfidence = 0.4;
                    finalType = 'CHỐNG ĐẢO (SAU ' + stats.consecutiveLosses + ' LẦN THUA)';
                    finalPattern = '';
                    finalReason = 'Chống đảo do thua liên tiếp';
                }

                // Lưu dự đoán cho phiên tiếp theo
                lastPrediction = {
                    phien: currentSessionId ? parseInt(currentSessionId) + 1 : null,
                    ket_qua: finalPrediction,
                    loai_cau: finalType,
                    mau_cau: finalPattern,
                    do_tin_cay: (finalConfidence * 100).toFixed(0) + '%'
                };

                // Trạng thái
                const trangThai = finalType.includes('CHỐNG') ? 'Chống đảo' :
                                 (finalType.includes('THEO') ? 'Đang theo kết quả' : 'Đang theo cầu');

                // Tỉ lệ
                const tiLe = stats.total > 0 ? ((stats.correct / stats.total) * 100).toFixed(1) + '%' : '0%';
                const neuralTiLe = stats.neuralTotal > 0 ? 
                    ((stats.neuralCorrect / stats.neuralTotal) * 100).toFixed(1) + '%' : '0%';

                // Cập nhật API - ID đã thay đổi
                apiResponseData = {
                    "Phien": currentSessionId,
                    "Xuc_xac_1": d1,
                    "Xuc_xac_2": d2,
                    "Xuc_xac_3": d3,
                    "Tong": total,
                    "Ket_qua": result,
                    "Phien_hien_tai": currentSessionId ? parseInt(currentSessionId) + 1 : null,
                    "Du_doan": finalPrediction,
                    "Loai_cau": finalType,
                    "Mau_cau_phat_hien": finalPattern,
                    "Do_tin_cay": (finalConfidence * 100).toFixed(0) + '%',
                    "Trang_thai": trangThai,
                    "Ket_qua_du_doan": predictionCorrect ? '✅' : (stats.total > 0 ? '❌' : ''),
                    "Thong_ke": {
                        "tong": stats.total,
                        "dung": stats.correct,
                        "sai": stats.wrong,
                        "ti_le": tiLe,
                        "neural_ti_le": neuralTiLe
                    },
                    "id": "@tranhoang2286"
                };

                // Log
                console.log('\n' + '🟦'.repeat(25));
                console.log(`🎲 Phiên ${apiResponseData.Phien} | KQ: ${result} | Tổng: ${total}`);
                console.log(`📊 Lịch sử: ${historyForAnalyzer.slice(-12).map(h => h.Ket_qua).join(' ')}`);
                console.log(`🔍 Phát hiện: ${finalType} | Mẫu: ${finalPattern || '...'}`);
                console.log(`🤖 Dự đoán phiên ${apiResponseData.Phien_hien_tai}: ${finalPrediction} (${(finalConfidence * 100).toFixed(0)}%)`);
                console.log(`📊 ${ensembleResult.details.length} models tham gia | Top: ${ensembleResult.details.slice(0,3).map(d => d.model).join(', ')}`);
                if (ensembleResult.neuralOutput) {
                    console.log(`🧠 Neural: ${ensembleResult.neuralOutput.prediction} (${(ensembleResult.neuralOutput.confidence*100).toFixed(0)}%) | Acc: ${neuralTiLe}`);
                }
                console.log(`📈 Thống kê: Đúng ${stats.correct}/${stats.total} (${tiLe}) ${apiResponseData.Ket_qua_du_doan}`);
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
        console.log(`[🔌] WebSocket closed. Code: ${code}, Reason: ${reason.toString()}`);
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
            consecutive_losses: stats.consecutiveLosses,
            neural_total: stats.neuralTotal,
            neural_correct: stats.neuralCorrect,
            neural_ti_le: stats.neuralTotal > 0 ? ((stats.neuralCorrect / stats.neuralTotal) * 100).toFixed(1) + '%' : '0%'
        }
    });
});

app.get('/api/models', (req, res) => {
    res.json({
        main_models: Object.keys(analyzer.modelWeights).length,
        sub_models: Object.keys(analyzer.subModels).length,
        mini_models: Object.keys(analyzer.miniModels).length,
        total: 21 + 42 + 21,
        neural_network: '84→42→2 (ReLU + Softmax)',
        weights: {
            main: analyzer.modelWeights,
            sub: analyzer.subModelWeights,
            mini: analyzer.miniModelWeights
        },
        neural_stats: {
            total: stats.neuralTotal,
            correct: stats.neuralCorrect,
            accuracy: stats.neuralTotal > 0 ? ((stats.neuralCorrect / stats.neuralTotal) * 100).toFixed(1) + '%' : '0%'
        }
    });
});

app.get('/', (req, res) => {
    res.json(apiResponseData);
});

app.listen(PORT, () => {
    console.log(`[🌐] Server is running at http://localhost:${PORT}`);
    console.log(`[📁] History file: ${HISTORY_FILE}`);
    console.log(`[📁] Patterns file: ${PATTERNS_FILE}`);
    console.log(`[📁] Model weights file: ${MODEL_WEIGHTS_FILE}`);
    console.log(`[🤖] Total models: 21 main + 42 sub + 21 mini = 84 models`);
    console.log(`[🧠] Neural Network: 84→42→2 (ReLU + Softmax + CrossEntropy + Momentum)`);
    console.log(`[🧠] Sub models có tư duy riêng về từng loại cầu:`);
    console.log(`     - Model 1-6: Chuyên cầu 1-1`);
    console.log(`     - Model 7-12: Chuyên cầu 2-2`);
    console.log(`     - Model 13-18: Chuyên cầu bệt`);
    console.log(`     - Model 19-24: Chuyên cầu 3-3`);
    console.log(`     - Model 25-30: Chuyên cầu 2-1-2 và 1-2-1`);
    console.log(`     - Model 31-36: Chuyên bẻ cầu và chuyển tiếp`);
    console.log(`     - Model 37-42: Chuyên phân tích tổng hợp`);
    console.log(`[🆔] ID: @tranhoang2286`);
});

// ==================== START ====================
connectWebSocket();