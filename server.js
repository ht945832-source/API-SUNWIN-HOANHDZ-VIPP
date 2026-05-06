// ==================== FIX NODE.JS ====================
process.on('uncaughtException', (err) console.error('Exception:', err));
process.on('unhandledRejection', (reason) => console.error('Rejection:', reason));

const dns = require('dns');
dns.setDefaultResultOrder('ipv4first');

console.log(`✅ Node.js: ${process.version}`);

// ==================== IMPORTS ====================
const WebSocket = require('ws');
const express = require('express');
const cors = require('cors');
const fs = require('fs');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 10000;

// ==================== FILE STORAGE ====================
const HISTORY_FILE = './history.json';
const PATTERN_LIBRARY_FILE = './pattern_library.json';
const AI_WEIGHTS_FILE = './ai_weights.json';

let resultHistory = [];
let patternLibrary = {};
let aiWeights = {};

// Load files
try { resultHistory = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8') || '[]'); } catch(e) {}
try { patternLibrary = JSON.parse(fs.readFileSync(PATTERN_LIBRARY_FILE, 'utf8') || '{}'); } catch(e) {}
try { aiWeights = JSON.parse(fs.readFileSync(AI_WEIGHTS_FILE, 'utf8') || '{}'); } catch(e) {}

// ==================== PHÂN TÍCH CẦU THÔNG MINH ====================
class UltimatePatternAnalyzer {
    constructor() {
        // 15 loại cầu khác nhau
        this.patternTypes = {
            '1-1': { weight: 1.0, detected: false, confidence: 0 },
            '2-2': { weight: 1.0, detected: false, confidence: 0 },
            '3-3': { weight: 1.0, detected: false, confidence: 0 },
            '1-2-1': { weight: 1.0, detected: false, confidence: 0 },
            '2-1-2': { weight: 1.0, detected: false, confidence: 0 },
            '1-1-2-2': { weight: 1.0, detected: false, confidence: 0 },
            '2-2-1-1': { weight: 1.0, detected: false, confidence: 0 },
            '1-2-2-1': { weight: 1.0, detected: false, confidence: 0 },
            '2-1-1-2': { weight: 1.0, detected: false, confidence: 0 },
            'bệt': { weight: 1.0, detected: false, confidence: 0 },
            'xen_kẽ': { weight: 1.0, detected: false, confidence: 0 },
            'giảm_dần': { weight: 1.0, detected: false, confidence: 0 },
            'tăng_dần': { weight: 1.0, detected: false, confidence: 0 },
            'đối_xứng': { weight: 1.0, detected: false, confidence: 0 },
            'chu_kỳ': { weight: 1.0, detected: false, confidence: 0 }
        };
        
        // Điểm số cho từng loại dự đoán
        this.predictionScores = { Tai: 0, Xiu: 0 };
        this.totalWeight = 0;
    }
    
    // ========== 1. PHÂN TÍCH CẦU 1-1 (XEN KẼ HOÀN HẢO) ==========
    analyzePattern11(results) {
        if (results.length < 4) return null;
        
        const last6 = results.slice(-6);
        let isPerfect = true;
        
        for (let i = 1; i < last6.length; i++) {
            if (last6[i] === last6[i-1]) {
                isPerfect = false;
                break;
            }
        }
        
        if (isPerfect && last6.length >= 4) {
            const last = results[results.length - 1];
            const prediction = last === 'Tài' ? 'Xỉu' : 'Tài';
            let confidence = 0.75;
            
            // Tăng độ tin cậy nếu cầu càng dài
            if (last6.length >= 6) confidence = 0.85;
            if (last6.length >= 8) confidence = 0.92;
            
            return {
                pattern: '1-1',
                prediction: prediction,
                confidence: confidence,
                description: `Cầu 1-1 chuẩn ${last6.length} phiên`,
                nextMove: prediction
            };
        }
        
        // Cầu 1-1 biến thể (có 1 lỗi)
        let errorCount = 0;
        for (let i = 1; i < last6.length; i++) {
            if (last6[i] === last6[i-1]) errorCount++;
        }
        
        if (errorCount === 1 && last6.length >= 5) {
            const last = results[results.length - 1];
            const prediction = last === 'Tài' ? 'Xỉu' : 'Tài';
            return {
                pattern: '1-1',
                prediction: prediction,
                confidence: 0.68,
                description: 'Cầu 1-1 biến thể (1 lỗi)',
                nextMove: prediction
            };
        }
        
        return null;
    }
    
    // ========== 2. PHÂN TÍCH CẦU 2-2 (CẶP ĐÔI) ==========
    analyzePattern22(results) {
        if (results.length < 6) return null;
        
        const last8 = results.slice(-8);
        
        // Pattern 2-2 chuẩn: TTXXTTXX hoặc XXTTXXTT
        let isPattern22 = true;
        for (let i = 0; i < last8.length - 2; i += 2) {
            if (last8[i] !== last8[i+1]) {
                isPattern22 = false;
                break;
            }
            if (i + 2 < last8.length && last8[i] === last8[i+2]) {
                isPattern22 = false;
                break;
            }
        }
        
        if (isPattern22 && last8.length >= 6) {
            const last = results[results.length - 1];
            const nextInPair = last;
            let prediction = nextInPair;
            let confidence = 0.78;
            
            // Nếu đã có 2 cặp, dự đoán cặp tiếp theo
            if (last8.length >= 8) {
                const firstOfLastPair = last8[last8.length - 2];
                prediction = firstOfLastPair;
                confidence = 0.82;
            }
            
            return {
                pattern: '2-2',
                prediction: prediction,
                confidence: confidence,
                description: 'Cầu 2-2 (cặp đôi)',
                nextMove: prediction
            };
        }
        
        return null;
    }
    
    // ========== 3. PHÂN TÍCH CẦU 3-3 ==========
    analyzePattern33(results) {
        if (results.length < 9) return null;
        
        const last12 = results.slice(-12);
        
        // Pattern 3-3: TTTXXXTTTXXX
        let isPattern33 = true;
        for (let i = 0; i < last12.length - 3; i += 3) {
            // Kiểm tra 3 phiên giống nhau
            if (last12[i] !== last12[i+1] || last12[i+1] !== last12[i+2]) {
                isPattern33 = false;
                break;
            }
            // Kiểm tra khác với bộ trước
            if (i + 3 < last12.length && last12[i] === last12[i+3]) {
                isPattern33 = false;
                break;
            }
        }
        
        if (isPattern33 && last12.length >= 9) {
            const last = results[results.length - 1];
            const firstOfCurrentBlock = last12[Math.floor((last12.length - 3) / 3) * 3];
            const prediction = firstOfCurrentBlock === 'Tài' ? 'Xỉu' : 'Tài';
            
            return {
                pattern: '3-3',
                prediction: prediction,
                confidence: 0.85,
                description: 'Cầu 3-3 siêu chuẩn',
                nextMove: prediction
            };
        }
        
        return null;
    }
    
    // ========== 4. PHÂN TÍCH CẦU 1-2-1 ==========
    analyzePattern121(results) {
        if (results.length < 5) return null;
        
        const last5 = results.slice(-5);
        
        // Pattern: T X X T ? hoặc X T T X ?
        if (last5[0] !== last5[1] && last5[1] === last5[2] && last5[2] !== last5[3] && last5[3] === last5[4]) {
            const prediction = last5[4];
            return {
                pattern: '1-2-1',
                prediction: prediction,
                confidence: 0.75,
                description: 'Cầu 1-2-1 hoàn hảo',
                nextMove: prediction
            };
        }
        
        // Pattern mở rộng: 1-2-1-2-1
        if (results.length >= 7) {
            const last7 = results.slice(-7);
            if (last7[0] !== last7[1] && last7[1] === last7[2] && last7[2] !== last7[3] &&
                last7[3] === last7[4] && last7[4] !== last7[5] && last7[5] === last7[6]) {
                const prediction = last7[6];
                return {
                    pattern: '1-2-1',
                    prediction: prediction,
                    confidence: 0.82,
                    description: 'Cầu 1-2-1-2-1 mở rộng',
                    nextMove: prediction
                };
            }
        }
        
        return null;
    }
    
    // ========== 5. PHÂN TÍCH CẦU 2-1-2 ==========
    analyzePattern212(results) {
        if (results.length < 5) return null;
        
        const last5 = results.slice(-5);
        
        // Pattern: T T X T T ? hoặc X X T X X ?
        if (last5[0] === last5[1] && last5[1] !== last5[2] && last5[2] !== last5[3] && last5[3] === last5[4]) {
            const firstOfLastPair = last5[3];
            const prediction = firstOfLastPair;
            return {
                pattern: '2-1-2',
                prediction: prediction,
                confidence: 0.78,
                description: 'Cầu 2-1-2 chuẩn',
                nextMove: prediction
            };
        }
        
        return null;
    }
    
    // ========== 6. PHÂN TÍCH CẦU BỆT (DÂY) ==========
    analyzeStreak(results) {
        if (results.length < 3) return null;
        
        let streak = 1;
        const last = results[results.length - 1];
        
        for (let i = results.length - 2; i >= 0; i--) {
            if (results[i] === last) streak++;
            else break;
        }
        
        if (streak >= 3) {
            let confidence = 0.55;
            let description = `Bệt ${streak} phiên`;
            
            if (streak >= 5) {
                confidence = 0.72;
                description = `Bệt dài ${streak} phiên, có thể sắp gãy`;
            }
            if (streak >= 7) {
                confidence = 0.45;
                description = `Siêu bệt ${streak} phiên, khả năng đảo chiều cao`;
                const opposite = last === 'Tài' ? 'Xỉu' : 'Tài';
                return {
                    pattern: 'bệt',
                    prediction: opposite,
                    confidence: confidence,
                    description: description,
                    nextMove: opposite
                };
            }
            
            return {
                pattern: 'bệt',
                prediction: last,
                confidence: confidence,
                description: description,
                nextMove: last
            };
        }
        
        return null;
    }
    
    // ========== 7. PHÂN TÍCH CẦU XEN KẼ HOÀN HẢO ==========
    analyzePerfectAlternate(results) {
        if (results.length < 8) return null;
        
        const last10 = results.slice(-10);
        let alternateCount = 0;
        
        for (let i = 1; i < last10.length; i++) {
            if (last10[i] !== last10[i-1]) alternateCount++;
        }
        
        const alternateRatio = alternateCount / (last10.length - 1);
        
        if (alternateRatio >= 0.8) {
            const last = results[results.length - 1];
            const prediction = last === 'Tài' ? 'Xỉu' : 'Tài';
            const confidence = 0.7 + (alternateRatio - 0.8) * 1.5;
            
            return {
                pattern: 'xen_kẽ',
                prediction: prediction,
                confidence: Math.min(confidence, 0.88),
                description: `Cầu xen kẽ ${(alternateRatio * 100).toFixed(0)}%`,
                nextMove: prediction
            };
        }
        
        return null;
    }
    
    // ========== 8. PHÂN TÍCH CẦU GIẢM DẦN / TĂNG DẦN ==========
    analyzeTrend(results) {
        if (results.length < 6) return null;
        
        const last6 = results.slice(-6);
        let taiCount = 0, xiuCount = 0;
        
        for (let i = 0; i < last6.length; i++) {
            if (last6[i] === 'Tài') taiCount++;
            else xiuCount++;
        }
        
        // Giảm dần Tài
        if (taiCount === 5) {
            return {
                pattern: 'giảm_dần',
                prediction: 'Xỉu',
                confidence: 0.68,
                description: 'Tài đang giảm dần, dự đoán Xỉu',
                nextMove: 'Xỉu'
            };
        }
        
        // Giảm dần Xỉu
        if (xiuCount === 5) {
            return {
                pattern: 'giảm_dần',
                prediction: 'Tài',
                confidence: 0.68,
                description: 'Xỉu đang giảm dần, dự đoán Tài',
                nextMove: 'Tài'
            };
        }
        
        // Tăng dần
        const last3 = last6.slice(-3);
        const prev3 = last6.slice(0, 3);
        
        const last3Tai = last3.filter(r => r === 'Tài').length;
        const prev3Tai = prev3.filter(r => r === 'Tài').length;
        
        if (last3Tai > prev3Tai + 1) {
            return {
                pattern: 'tăng_dần',
                prediction: 'Tài',
                confidence: 0.62,
                description: 'Xu hướng Tài tăng dần',
                nextMove: 'Tài'
            };
        }
        
        if (last3Tai < prev3Tai - 1) {
            return {
                pattern: 'tăng_dần',
                prediction: 'Xỉu',
                confidence: 0.62,
                description: 'Xu hướng Xỉu tăng dần',
                nextMove: 'Xỉu'
            };
        }
        
        return null;
    }
    
    // ========== 9. PHÂN TÍCH CẦU ĐỐI XỨNG ==========
    analyzeSymmetry(results) {
        if (results.length < 7) return null;
        
        // Kiểm tra đối xứng qua tâm
        for (let radius = 2; radius <= 4; radius++) {
            if (results.length < radius * 2) continue;
            
            const left = results.slice(results.length - radius * 2, results.length - radius);
            const right = results.slice(results.length - radius, results.length);
            
            let symmetric = true;
            for (let i = 0; i < radius; i++) {
                if (left[i] !== right[radius - 1 - i]) {
                    symmetric = false;
                    break;
                }
            }
            
            if (symmetric) {
                const prediction = right[radius - 1];
                const confidence = 0.7 + (radius * 0.03);
                
                return {
                    pattern: 'đối_xứng',
                    prediction: prediction,
                    confidence: Math.min(confidence, 0.85),
                    description: `Cầu đối xứng bán kính ${radius}`,
                    nextMove: prediction
                };
            }
        }
        
        return null;
    }
    
    // ========== 10. PHÂN TÍCH CHU KỲ ==========
    analyzeCycle(results) {
        if (results.length < 8) return null;
        
        // Tìm chu kỳ từ 2-5 phiên
        for (let cycleLen = 2; cycleLen <= 5; cycleLen++) {
            if (results.length < cycleLen * 2) continue;
            
            const lastCycle = results.slice(-cycleLen);
            let matchCount = 0;
            
            for (let i = 0; i <= results.length - cycleLen * 2; i += cycleLen) {
                let match = true;
                for (let j = 0; j < cycleLen; j++) {
                    if (results[i + j] !== lastCycle[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) matchCount++;
            }
            
            if (matchCount >= 2) {
                const prediction = lastCycle[0];
                const confidence = 0.6 + (matchCount * 0.05) + (cycleLen * 0.02);
                
                return {
                    pattern: 'chu_kỳ',
                    prediction: prediction,
                    confidence: Math.min(confidence, 0.85),
                    description: `Chu kỳ ${cycleLen} phiên lặp ${matchCount} lần`,
                    nextMove: prediction
                };
            }
        }
        
        return null;
    }
    
    // ========== 11. PHÂN TÍCH PATTERN 1-1-2-2 ==========
    analyzePattern1122(results) {
        if (results.length < 6) return null;
        
        const last6 = results.slice(-6);
        
        // Pattern: T X T T X X ?
        if (last6[0] !== last6[1] && last6[1] !== last6[2] && 
            last6[2] === last6[3] && last6[3] !== last6[4] && last6[4] === last6[5]) {
            const prediction = last6[5];
            return {
                pattern: '1-1-2-2',
                prediction: prediction,
                confidence: 0.72,
                description: 'Cầu 1-1-2-2',
                nextMove: prediction
            };
        }
        
        return null;
    }
    
    // ========== 12. PHÂN TÍCH PATTERN 2-2-1-1 ==========
    analyzePattern2211(results) {
        if (results.length < 6) return null;
        
        const last6 = results.slice(-6);
        
        // Pattern: T T X T T X ?
        if (last6[0] === last6[1] && last6[1] !== last6[2] && 
            last6[2] !== last6[3] && last6[3] === last6[4] && last6[4] !== last6[5]) {
            const prediction = last6[5];
            return {
                pattern: '2-2-1-1',
                prediction: prediction,
                confidence: 0.72,
                description: 'Cầu 2-2-1-1',
                nextMove: prediction
            };
        }
        
        return null;
    }
    
    // ========== 13. PHÂN TÍCH DỰA TRÊN TỔNG ĐIỂM XÚC XẮC ==========
    analyzeScorePattern(history) {
        if (!history || history.length < 5) return null;
        
        const scores = history.slice(-10).map(h => h.Tong).filter(s => s);
        if (scores.length < 5) return null;
        
        // Tính trung bình và độ lệch
        const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
        const lastScore = scores[scores.length - 1];
        
        // Điểm cao bất thường -> dự đoán Xỉu
        if (lastScore >= 15 && avg <= 11) {
            return {
                pattern: 'điểm_cao',
                prediction: 'Xỉu',
                confidence: 0.68,
                description: `Điểm ${lastScore} cao bất thường`,
                nextMove: 'Xỉu'
            };
        }
        
        // Điểm thấp bất thường -> dự đoán Tài
        if (lastScore <= 6 && avg >= 10) {
            return {
                pattern: 'điểm_thấp',
                prediction: 'Tài',
                confidence: 0.68,
                description: `Điểm ${lastScore} thấp bất thường`,
                nextMove: 'Tài'
            };
        }
        
        // Xu hướng điểm
        let trend = 0;
        for (let i = 1; i < scores.length; i++) {
            trend += scores[i] - scores[i-1];
        }
        
        if (trend > 5) {
            return {
                pattern: 'xu_hướng',
                prediction: 'Xỉu',
                confidence: 0.58,
                description: 'Điểm đang tăng nhanh',
                nextMove: 'Xỉu'
            };
        }
        
        if (trend < -5) {
            return {
                pattern: 'xu_hướng',
                prediction: 'Tài',
                confidence: 0.58,
                description: 'Điểm đang giảm nhanh',
                nextMove: 'Tài'
            };
        }
        
        return null;
    }
    
    // ========== 14. MA TRẬN CHUYỂN TIẾP THÔNG MINH ==========
    analyzeTransitionMatrix(results) {
        if (results.length < 15) return null;
        
        // Xây dựng ma trận chuyển tiếp 2 cấp
        const transitions = {};
        const pairs = ['TT', 'TX', 'XT', 'XX'];
        
        pairs.forEach(p => transitions[p] = { T: 0, X: 0 });
        
        for (let i = 2; i < results.length; i++) {
            const key = results[i-2] === 'Tài' ? 'T' : 'X';
            const prev = results[i-1] === 'Tài' ? 'T' : 'X';
            const current = results[i] === 'Tài' ? 'T' : 'X';
            const transitionKey = key + prev;
            
            if (transitions[transitionKey]) {
                transitions[transitionKey][current]++;
            }
        }
        
        // Lấy 2 kết quả gần nhất
        if (results.length >= 2) {
            const last1 = results[results.length - 1] === 'Tài' ? 'T' : 'X';
            const last2 = results[results.length - 2] === 'Tài' ? 'T' : 'X';
            const key = last2 + last1;
            
            const transition = transitions[key];
            if (transition && (transition.T + transition.X) >= 3) {
                const taiProb = transition.T / (transition.T + transition.X);
                const xiuProb = transition.X / (transition.T + transition.X);
                
                if (taiProb > 0.65) {
                    return {
                        pattern: 'ma_trận',
                        prediction: 'Tài',
                        confidence: 0.6 + taiProb * 0.1,
                        description: `Ma trận: ${key} → Tài (${(taiProb*100).toFixed(0)}%)`,
                        nextMove: 'Tài'
                    };
                }
                
                if (xiuProb > 0.65) {
                    return {
                        pattern: 'ma_trận',
                        prediction: 'Xỉu',
                        confidence: 0.6 + xiuProb * 0.1,
                        description: `Ma trận: ${key} → Xỉu (${(xiuProb*100).toFixed(0)}%)`,
                        nextMove: 'Xỉu'
                    };
                }
            }
        }
        
        return null;
    }
    
    // ========== 15. PHÁT HIỆN PATTERN ĐẶC BIỆT TỪ THƯ VIỆN ==========
    analyzePatternLibrary(results) {
        if (results.length < 5 || Object.keys(patternLibrary).length === 0) return null;
        
        const last5 = results.slice(-5).join('');
        
        for (const [pattern, data] of Object.entries(patternLibrary)) {
            if (pattern === last5 && data.next && data.confidence > 0.6) {
                return {
                    pattern: 'thư_viện',
                    prediction: data.next,
                    confidence: data.confidence,
                    description: `Pattern lưu trữ: ${pattern}`,
                    nextMove: data.next
                };
            }
        }
        
        return null;
    }
    
    // ========== TỔNG HỢP TẤT CẢ CÁC PHÂN TÍCH ==========
    analyzeAll(history) {
        const results = history.map(h => h.Ket_qua).filter(r => r);
        
        if (results.length < 5) {
            return {
                prediction: 'Tài',
                confidence: 0.5,
                pattern: 'chưa_đủ_dữ_liệu',
                description: 'Đang thu thập dữ liệu...',
                details: []
            };
        }
        
        const analyses = [];
        
        // Chạy tất cả các phương pháp phân tích
        const methods = [
            this.analyzePattern11.bind(this),
            this.analyzePattern22.bind(this),
            this.analyzePattern33.bind(this),
            this.analyzePattern121.bind(this),
            this.analyzePattern212.bind(this),
            this.analyzePattern1122.bind(this),
            this.analyzePattern2211.bind(this),
            this.analyzeStreak.bind(this),
            this.analyzePerfectAlternate.bind(this),
            this.analyzeTrend.bind(this),
            this.analyzeSymmetry.bind(this),
            this.analyzeCycle.bind(this),
            this.analyzeScorePattern.bind(this),
            this.analyzeTransitionMatrix.bind(this),
            this.analyzePatternLibrary.bind(this)
        ];
        
        for (const method of methods) {
            const result = method(results);
            if (result) {
                analyses.push(result);
            }
        }
        
        if (analyses.length === 0) {
            // Fallback: phân tích đơn giản
            const last5 = results.slice(-5);
            const taiCount = last5.filter(r => r === 'Tài').length;
            const prediction = taiCount >= 3 ? 'Xỉu' : 'Tài';
            
            return {
                prediction: prediction,
                confidence: 0.52,
                pattern: 'cơ_bản',
                description: 'Phân tích cơ bản',
                details: []
            };
        }
        
        // Tính điểm từ các phân tích
        let taiScore = 0, xiuScore = 0, totalWeight = 0;
        const bestAnalyses = analyses.sort((a, b) => b.confidence - a.confidence).slice(0, 5);
        
        for (const analysis of bestAnalyses) {
            const weight = analysis.confidence;
            if (analysis.prediction === 'Tài') {
                taiScore += weight;
            } else {
                xiuScore += weight;
            }
            totalWeight += weight;
        }
        
        const finalPrediction = taiScore > xiuScore ? 'Tài' : 'Xỉu';
        let finalConfidence = Math.max(taiScore, xiuScore) / totalWeight;
        
        // Đảm bảo confidence trong khoảng 48% - 92%
        finalConfidence = Math.max(finalConfidence, 0.48);
        finalConfidence = Math.min(finalConfidence, 0.92);
        
        // Chọn pattern tốt nhất để hiển thị
        const bestPattern = bestAnalyses[0];
        
        return {
            prediction: finalPrediction,
            confidence: finalConfidence,
            pattern: bestPattern.pattern,
            description: bestPattern.description,
            details: bestAnalyses.map(a => `${a.pattern}: ${a.prediction} (${(a.confidence*100).toFixed(0)}%)`),
            allAnalyses: analyses.length
        };
    }
    
    // Lưu pattern vào thư viện
    savePattern(results, prediction, confidence, actual) {
        if (results.length < 5) return;
        
        const last5 = results.slice(-5).join('');
        const isCorrect = prediction === actual;
        
        if (!patternLibrary[last5]) {
            patternLibrary[last5] = { next: prediction, count: 0, correct: 0, confidence: 0 };
        }
        
        patternLibrary[last5].count++;
        if (isCorrect) patternLibrary[last5].correct++;
        patternLibrary[last5].confidence = patternLibrary[last5].correct / patternLibrary[last5].count;
        patternLibrary[last5].next = patternLibrary[last5].confidence > 0.55 ? prediction : 
                                      (patternLibrary[last5].next === 'Tài' ? 'Xỉu' : 'Tài');
        
        // Giới hạn kích thước thư viện
        const entries = Object.entries(patternLibrary);
        if (entries.length > 200) {
            const sorted = entries.sort((a, b) => b[1].count - a[1].count);
            patternLibrary = Object.fromEntries(sorted.slice(0, 150));
        }
        
        fs.writeFileSync(PATTERN_LIBRARY_FILE, JSON.stringify(patternLibrary, null, 2));
    }
}

// ==================== GLOBAL VARIABLES ====================
const analyzer = new UltimatePatternAnalyzer();
let currentSessionId = null;
let lastPrediction = null;

let stats = {
    total: 0,
    correct: 0,
    wrong: 0,
    consecutiveLosses: 0,
    patternStats: {}
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
    "Phan_tich_chi_tiet": [],
    "Thong_ke": { "tong": 0, "dung": 0, "sai": 0, "ti_le": "0%" },
    "id": "@tranhoang2286"
};

// ==================== WEBSOCKET ====================
const WS_URL = "wss://websocket.azhkthg1.net/websocket?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhbW91bnQiOjAsInVzZXJuYW1lIjoiU0NfYXBpc3Vud2luMTIzIn0.hgrRbSV6vnBwJMg9ZFtbx3rRu9mX_hZMZ_m5gMNhkw0";

let ws = null;
let reconnectAttempts = 0;

const INIT_MSGS = [
    [1, "MiniGame", "GM_apivopnha", "WangLin", { "info": "{\"ipAddress\":\"14.249.227.107\"}" }],
    [6, "MiniGame", "taixiuPlugin", { cmd: 1005 }],
    [6, "MiniGame", "lobbyPlugin", { cmd: 10001 }]
];

function connectWS() {
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    
    console.log(`[🔄] Kết nối WS (lần ${reconnectAttempts + 1})...`);
    
    try {
        ws = new WebSocket(WS_URL);
        
        ws.on('open', () => {
            console.log('[✅] WebSocket connected!');
            reconnectAttempts = 0;
            
            INIT_MSGS.forEach((msg, i) => {
                setTimeout(() => {
                    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
                }, i * 600);
            });
            
            setInterval(() => {
                if (ws?.readyState === WebSocket.OPEN) ws.ping();
            }, 25000);
        });
        
        ws.on('message', (data) => {
            try {
                const msg = JSON.parse(data.toString());
                if (!Array.isArray(msg) || !msg[1]) return;
                
                const { cmd, sid, d1, d2, d3 } = msg[1];
                
                if (cmd === 1008 && sid) currentSessionId = sid;
                
                if (cmd === 1003 && d1 && d2 && d3) {
                    const total = d1 + d2 + d3;
                    const result = total > 10 ? "Tài" : "Xỉu";
                    
                    let correct = false;
                    if (lastPrediction) {
                        correct = lastPrediction.ket_qua === result;
                        stats.total++;
                        if (correct) {
                            stats.correct++;
                            stats.consecutiveLosses = 0;
                        } else {
                            stats.wrong++;
                            stats.consecutiveLosses++;
                        }
                        
                        // Lưu pattern
                        const historyResults = resultHistory.slice(-10).map(h => h.Ket_qua);
                        analyzer.savePattern(historyResults, lastPrediction.ket_qua, 
                            parseFloat(lastPrediction.do_tin_cay) / 100, result);
                    }
                    
                    // Lưu lịch sử
                    resultHistory.push({
                        phien: currentSessionId,
                        Xuc_xac: [d1, d2, d3],
                        Tong: total,
                        Ket_qua: result,
                        time: Date.now()
                    });
                    if (resultHistory.length > 300) resultHistory.shift();
                    fs.writeFileSync(HISTORY_FILE, JSON.stringify(resultHistory, null, 2));
                    
                    // Phân tích
                    const analysis = analyzer.analyzeAll(resultHistory);
                    let finalPrediction = analysis.prediction;
                    let finalConfidence = analysis.confidence;
                    
                    // Chống đảo
                    if (stats.consecutiveLosses >= 3) {
                        finalPrediction = finalPrediction === 'Tài' ? 'Xỉu' : 'Tài';
                        finalConfidence = 0.48;
                    }
                    
                    lastPrediction = {
                        phien: currentSessionId ? parseInt(currentSessionId) + 1 : null,
                        ket_qua: finalPrediction,
                        loai_cau: analysis.pattern,
                        do_tin_cay: (finalConfidence * 100).toFixed(0) + '%'
                    };
                    
                    const tiLe = stats.total > 0 ? ((stats.correct / stats.total) * 100).toFixed(1) + '%' : '0%';
                    
                    apiResponseData = {
                        "Phien": currentSessionId,
                        "Xuc_xac_1": d1, "Xuc_xac_2": d2, "Xuc_xac_3": d3,
                        "Tong": total,
                        "Ket_qua": result,
                        "Phien_hien_tai": currentSessionId ? parseInt(currentSessionId) + 1 : null,
                        "Du_doan": finalPrediction,
                        "Loai_cau": analysis.pattern,
                        "Mau_cau_phat_hien": analysis.description,
                        "Do_tin_cay": (finalConfidence * 100).toFixed(0) + '%',
                        "Phan_tich_chi_tiet": analysis.details || [],
                        "Thong_ke": { "tong": stats.total, "dung": stats.correct, "sai": stats.wrong, "ti_le": tiLe },
                        "id": "@tranhoang2286"
                    };
                    
                    console.log(`\n🎲 Phiên ${currentSessionId} | KQ: ${result}`);
                    console.log(`🔍 Phát hiện: ${analysis.pattern} | ${analysis.description}`);
                    console.log(`🎯 Dự đoán: ${finalPrediction} (${(finalConfidence*100).toFixed(0)}%) | ${correct ? '✅' : '❌'}`);
                    console.log(`📈 Thống kê: ${stats.correct}/${stats.total} (${tiLe})`);
                    console.log(`📊 Phân tích: ${analysis.details?.length || 0} patterns\n`);
                    
                    currentSessionId = null;
                }
            } catch (e) {}
        });
        
        ws.on('close', () => {
            console.log('[🔌] WS closed');
            reconnectAttempts++;
            setTimeout(connectWS, Math.min(5000, 1000 * reconnectAttempts));
        });
        
        ws.on('error', (err) => console.error('[❌] WS error:', err.message));
        
    } catch (err) {
        console.error('[❌] WS creation error:', err.message);
        setTimeout(connectWS, 3000);
    }
}

// ==================== API ENDPOINTS ====================
app.get('/api/ditmemaysun', (req, res) => res.json(apiResponseData));
app.get('/api/stats', (req, res) => res.json(stats));
app.get('/api/patterns', (req, res) => res.json(patternLibrary));
app.get('/api/history', (req, res) => res.json(resultHistory.slice(-30).reverse()));
app.get('/health', (req, res) => res.json({ status: 'ok', time: new Date().toISOString() }));
app.get('/', (req, res) => res.json({ name: 'Tài Xỉu API', version: '3.0', author: '@tranhoang2286' }));

// ==================== START ====================
app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n${'='.repeat(50)}`);
    console.log(`[🌐] Server: http://localhost:${PORT}`);
    console.log(`[🤖] 15+ phương pháp phân tích cầu`);
    console.log(`[📊] Độ tin cậy tối thiểu: 48%`);
    console.log(`[🆔] @tranhoang2286`);
    console.log(`${'='.repeat(50)}\n`);
    setTimeout(connectWS, 2000);
});