---
title: Cloudflare 配置指南
---

# Cloudflare Pages + Access 配置指南

本文介绍如何配置 Cloudflare Pages 自动部署和 Cloudflare Access 认证。

## 前置条件

- Cloudflare 账号（Free 计划即可，Access 免费支持 50 用户）
- 自定义域名已托管在 Cloudflare（可选，也可使用 `*.pages.dev` 域名）
- GitHub Organization 管理员权限

## 第一步：配置 Cloudflare Pages

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. 进入 **Workers & Pages** > **Create**
3. 选择 **Connect to Git**，关联 GitHub 仓库 `primatrix/wiki`
4. 配置构建设置：
   - **Production branch**: `main`
   - **Build command**: `npm run docs:build`
   - **Build output directory**: `docs/.vitepress/dist`
   - **Node.js version**: 在环境变量中设置 `NODE_VERSION=20`
5. 点击 **Save and Deploy**

## 第二步：配置自定义域名（可选）

1. 在 Pages 项目设置中，进入 **Custom domains**
2. 添加域名（如 `wiki.yourcompany.com`）
3. Cloudflare 会自动配置 DNS 和 SSL

## 第三步：配置 Cloudflare Access

1. 进入 **Zero Trust** > **Access** > **Applications**
2. 点击 **Add an application** > **Self-hosted**
3. 配置应用：
   - **Application name**: Internal Wiki
   - **Application domain**: 你的 wiki 域名
   - **Session Duration**: 24 hours
4. 配置 Access Policy：
   - **Policy name**: Org Members Only
   - **Action**: Allow
   - **Include**: Login Methods → GitHub
   - **Require**: GitHub Organizations → `你的组织名`
5. 保存应用

## 第四步：配置 GitHub OAuth App

1. Cloudflare Access 会引导你配置 GitHub OAuth
2. 进入 **Zero Trust** > **Settings** > **Authentication**
3. 添加 GitHub 作为 Identity Provider
4. 按提示在 GitHub 创建 OAuth App 并填入 Client ID 和 Secret

## 验证

1. 使用无痕窗口访问 wiki URL
2. 应看到 Cloudflare Access 登录页面
3. 点击 GitHub 登录，授权后应能看到 wiki 内容
4. 使用非组织成员的 GitHub 账号测试，应被拒绝访问

## 预览部署保护

Cloudflare Pages 会为每个 PR 生成预览 URL（`*.pages.dev`），这些 URL 默认**不受** Access 策略保护。保护方案：

- **方案 A**：在 Cloudflare Access 中创建第二个应用，使用通配符策略覆盖 `*.wiki-project.pages.dev`
- **方案 B**：将预览部署限制到一个已受 Access 保护的自定义域名
- **方案 C**：在 Cloudflare Pages 设置中禁用预览部署（最简单，适用于对未发布内容安全性要求高的场景）
