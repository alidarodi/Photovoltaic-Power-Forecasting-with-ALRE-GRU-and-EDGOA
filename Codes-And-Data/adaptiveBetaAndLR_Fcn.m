function stop = adaptiveBetaAndLR_Fcn(info, alpha, beta)
    persistent prevLoss prevGradMag initialLR
    
    stop = false;
    
    if isempty(prevLoss)
        prevLoss = inf;
        prevGradMag = 1.0;
        initialLR = 0.001;   % InitialLearnRate خودتان
        alpha = 0.01;        % اگر مقدار خاصی می‌خواهید، اینجا تنظیم کنید
        beta  = 0.005;
    end
    
    if strcmp(info.State, 'iteration')
        currentLoss = info.TrainingLoss;
        
        % ==================== تخمین پویای Gradient Magnitude ====================
        if isfield(info, 'GradientMagnitude') && ~isempty(info.GradientMagnitude)
            gradMag = info.GradientMagnitude;           % بهترین حالت (اگر موجود باشد)
        else
            % تخمین هوشمند بر اساس تغییرات loss + regularization
            if isinf(prevLoss)
                gradMag = 1.0;
            else
                lossChange = abs(currentLoss - prevLoss);
                % تخمین بر اساس سرعت تغییر loss (هرچه نوسان بیشتر، گرادیان بزرگ‌تر)
                gradMag = prevGradMag * 0.9 + 0.1 * (1 + 10*lossChange);
                gradMag = min(gradMag, 10);   % محدود کردن برای پایداری
            end
        end
        
        % ==================== Learning Rate Adaptive ====================
        if ~isinf(prevLoss)
            lossVariation = abs(currentLoss - prevLoss);
            
            % فرمول درخواستی شما
            adaptiveFactor = 1 + alpha * gradMag - beta * lossVariation;
            newLR = initialLR * adaptiveFactor;
            
            % محدود کردن LR در بازه منطقی
            newLR = max(1e-6, min(0.01, newLR));
            
        else
            newLR = initialLR;
        end
        
        % ==================== به‌روزرسانی مقادیر قبلی ====================
        prevLoss = currentLoss;
        prevGradMag = gradMag;
        
        
    end
end

