// mlp_residual.js —— MLP 残差推理引擎
(function() {
    window.MLPResidual = {
        weights: null,
        ready: false,

        async load(url) {
            const resp = await fetch(url);
            this.weights = await resp.json();
            this.ready = true;
            console.log("[MLPResidual] 权重加载完成");
        },

        // 输入: [tau, alpha_target, alpha_num, nu_num, omega0, zeta, dt]
        // 输出: 归一化残差 (需 × 160 得到角度)
        predict(x) {
            if (!this.ready) return 0.0;
            let a = x;
            for (let i = 0; i < this.weights.length; i++) {
                const { W, b } = this.weights[i];
                const z = new Array(b.length);
                for (let k = 0; k < b.length; k++) {
                    let s = b[k];
                    const Wk = W[k];
                    for (let j = 0; j < a.length; j++) s += Wk[j] * a[j];
                    z[k] = s;
                }
                // 最后一层线性，隐藏层 tanh
                if (i < this.weights.length - 1) {
                    for (let k = 0; k < z.length; k++) z[k] = Math.tanh(z[k]);
                }
                a = z;
            }
            return a[0];
        }
    };
})();