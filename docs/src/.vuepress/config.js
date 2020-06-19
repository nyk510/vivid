module.exports = {
  base: '/',
  port: 3000,
  title: 'VIVID',
  locales: {
    '/': {
      lang: 'ja',
      title: 'VIVID',
      description: 'Support Tools For Your Machine Learning Project Vividly',
    },
    '/en': {
      lang: 'en-US',
      title: 'VIVID',
      description: 'Support Tools For Your Machine Learning Project Vividly',
    }
  },
  themeConfig: {
    locales: {
      '/': {
        sidebar: {
          '/usage/': [
            {
              title: 'Usage',
              sidebarDepth: 2,
              collapsable: false,
              children: [
                '',
                'estimators',
                'customize',
                'utils',
                'features'
              ]
            }
          ]
        }
      },
      '/en': {
        sidebar: {
          '/usage/': [
            {
              title: 'Usage',
              collapsable: false,
              children: [
                '',
                'models',
                'customize',
                'features'
              ]
            }
          ]
        }
      }
    }
  },
  plugins: ['@vuepress/last-updated']
}
