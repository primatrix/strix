# Internal Wiki

内部技术知识库 — 基于 [VitePress](https://vitepress.dev/) 构建，部署在 [Cloudflare Pages](https://pages.cloudflare.com/) 上。

## 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run docs:dev

# 构建生产版本
npm run docs:build

# 预览构建结果
npm run docs:preview
```

## 如何贡献

1. 创建分支：`git checkout -b docs/your-topic`
2. 编写或修改 Markdown 文件（位于 `docs/` 目录下）
3. 本地预览确认无误：`npm run docs:dev`
4. 提交 PR，等待 Review

## 目录结构

- `docs/projects/` — 按项目组织的技术文档
- `docs/rfc/` — 技术决策记录 (RFC)
- `docs/best-practices/` — 工程最佳实践
- `docs/onboarding/` — 新人入门指南
- `docs/summaries/` — 技术总结与复盘报告

## 静态资源

- 共享图片放在 `docs/public/images/`
- 项目专属图片放在对应项目的 `docs/projects/<name>/assets/` 目录
- 图片大小建议不超过 500KB，优先使用 WebP 格式
