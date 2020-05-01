module.exports = {
  base: '/vivid/',
  port: 3000,
  title: 'VIVID',
  description: 'Support Tools For Your Machine Learning Project Vividly',
  themeConfig: {
    sidebar: {
      // '/': [
      //   {
      //     title: 'VIVID'
      //   }
      // ],
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
  },
  plugins: ['@vuepress/last-updated']
}
