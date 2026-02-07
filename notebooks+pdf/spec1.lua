return {
  -- the colorscheme should be available when starting Neovim
  {
    "neanias/everforest-nvim",
    lazy = false,
    priority = 1000,
    config = function()
	vim.g.everforest_background = "hard"
	vim.cmd("colorscheme everforest")
    end
  },

  {
    "nvim-neorg/neorg",
    -- lazy-load on filetype
    ft = "norg",
    -- options for neorg. This will automatically call `require("neorg").setup(opts)`
    opts = {
      load = {
        ["core.defaults"] = {},
      },
    },
  },

  {
    "hrsh7th/nvim-cmp",
    -- these dependencies will only be loaded when cmp loads
    -- dependencies are always lazy-loaded unless specified otherwise
    dependencies = {
      "hrsh7th/cmp-nvim-lsp",
      "hrsh7th/cmp-buffer",
      "neovim/nvim-lspconfig",
    },
    config = function()
      local cmp = require("cmp")
      vim.lsp.enable('pyright')
      vim.lsp.enable('tinymist')

      cmp.setup({
	completion = {
	  autocomplete = false
	},
	preselect = cmp.PreselectMode.None,
	mapping = cmp.mapping.preset.insert({
	  ['<Enter>'] = cmp.mapping.confirm({ select = true }),
	  ['<C-Space>'] = cmp.mapping.complete(),
	  ['<Tab>'] = cmp.mapping.select_next_item(),
	  ['<C-Tab>'] = cmp.mapping.select_prev_item()
	}),
      
	sources = cmp.config.sources({
	  { name = 'nvim_lsp' },
	  { name = 'buffer' }
        })
      })
    end,
  },

  {
    "nvim-treesitter/nvim-treesitter",
    build = ":TSUpdate",
    lazy = false,
    config = function()
      require("nvim-treesitter.config").setup({
	ensure_installed = { "python", "typst", "lua", "toml", "ninja", "rst", "markdown", "markdown_inline" },
	highlight = { enable = true },
	indent = { enable = true },
      })
    end,
  },

  {
    'chomosuke/typst-preview.nvim',
    lazy = false, -- or ft = 'typst'
    version = '1.*',
    opts = {}, -- lazy.nvim will implicitly calls `setup {}`
  },

  {
    "nvim-lualine/lualine.nvim",
    opts = { theme = "everforest" }
  },

  {
  "tpope/vim-sleuth",
  lazy = false
  }

}
