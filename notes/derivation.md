# Derivation Notes

<!-- 從這裡開始動筆。建議順序： -->

## 1. Forward process 的封閉解
<!-- 為什麼 q(x_t | x_0) 有 closed form：兩個高斯的合成 + 歸納，
     reparameterization 讓訓練可以一步跳到任意 t -->

## 2. ELBO 分解
<!-- log p(x0) 的變分下界如何拆成 L_T + sum L_{t-1} + L_0，
     每一項是兩個高斯之間的 KL -->

## 3. 從 KL 到 simplified loss
<!-- 後驗 q(x_{t-1}|x_t,x_0) 的均值參數化 → 預測 eps 的 MSE，
     Ho et al. 丟掉加權係數的理由 -->

## 4. DDIM：non-Markovian 視角
<!-- 同一個邊際分佈、不同的聯合分佈；eta 的角色；
     為什麼可以跳步、為什麼 eta=0 是確定性的 -->
