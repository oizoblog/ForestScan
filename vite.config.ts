import { defineConfig } from 'vite'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  base: process.env.GITHUB_PAGES ? `/${process.env.GITHUB_REPOSITORY?.split('/')[1] || 'ForestScan'}/` : '/',
  plugins: [VitePWA({
    registerType: 'autoUpdate',
    includeAssets: ['icon.svg', 'icon-192.png', 'icon-512.png'],
    manifest: {
      name: 'ForestScan 蘑菇扫描', short_name: 'ForestScan',
      description: '离线蘑菇疑似目标检测演示',
      start_url: '.', scope: '.', display: 'standalone',
      background_color: '#101b18', theme_color: '#101b18', lang: 'zh-CN',
      icons: [{ src: 'icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' }, { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' }]
    },
    workbox: {
      globPatterns: ['**/*.{js,css,html,svg,wasm}'],
      maximumFileSizeToCacheInBytes: 20 * 1024 * 1024,
      navigateFallback: 'index.html'
    }
  })],
  test: { environment: 'node' }
})
