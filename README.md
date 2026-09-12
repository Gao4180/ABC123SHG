# 个人财富管理配置器

基于 Streamlit 的个人资产配置工具：

- **模式一 · 我的资产优化**：输入资产代码或名称（支持 A 股/美股/ETF 混输），马科维茨均值-方差优化，KYC 风险测评定级自动约束风险档位，Black-Litterman 择时观点融合，肥尾蒙特卡洛前景模拟，历史情景压力测试，再平衡偏离提醒
- **模式二 · 财富方案推荐**：内置国内 ETF/QDII 池与美股 ETF 池（含产品风险等级、费率），自动生成稳健保本 / 经典 60-40 / 全天候 / 积极增长四套方案并横向对比，含择时信号与动态配置回测
- **模式三 · 多目标规划**：养老/教育/买房各自独立子组合，下滑轨道按期限定股票占比，自助法模拟目标达成概率

数据源级联：Yahoo Finance → 新浪 / 腾讯 → Alpha Vantage（可选 key）。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud（免费公网链接）

1. 把本目录推到你的 GitHub 仓库（Public）
2. 打开 https://share.streamlit.io ，用 GitHub 账号登录
3. 点 **New app** → 选择仓库 → Main file 填 `app.py` → Deploy
4. 几分钟后得到 `https://<你的应用名>.streamlit.app` 公开链接

可选：在 App settings → Secrets 中配置 `ALPHAVANTAGE_API_KEY = "你的key"`，
作为美股数据的最后备用源（https://www.alphavantage.co 免费注册）。

> 免责声明：本工具基于历史数据统计，不构成投资建议。
