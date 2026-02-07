-- Bootstrap lazy.nvim
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not (vim.uv or vim.loop).fs_stat(lazypath) then
  local lazyrepo = "https://github.com/folke/lazy.nvim.git"
  local out = vim.fn.system({ "git", "clone", "--filter=blob:none", "--branch=stable", lazyrepo, lazypath })
  if vim.v.shell_error ~= 0 then
    vim.api.nvim_echo({
      { "Failed to clone lazy.nvim:\n", "ErrorMsg" },
      { out, "WarningMsg" },
      { "\nPress any key to exit..." },
    }, true, {})
    vim.fn.getchar()
    os.exit(1)
  end
end
vim.opt.rtp:prepend(lazypath)

-- Make sure to setup `mapleader` and `maplocalleader` before
-- loading lazy.nvim so that mappings are correct.
-- This is also a good place to setup other settings (vim.opt)
vim.g.mapleader = " "
vim.g.maplocalleader = "\\"
vim.opt.clipboard = "unnamedplus"
vim.opt.shortmess:append("sI")
vim.diagnostic.config({
  virtual_text = {
    prefix = '=',
    -- source = "if_many"
  },
  signs = true,
  underline = true,
  update_in_insert = true,
  severity_sort = true,
})

-- Automatically export PDF on save using the LSP command
vim.api.nvim_create_autocmd("BufWritePost", {
  pattern = "*.typ",
  callback = function()
    -- 1. Check if Tinymist is attached so we don't crash
    local clients = vim.lsp.get_clients({ name = "tinymist" })
    if #clients == 0 then return end

    -- 2. Get the current file path
    local file_path = vim.api.nvim_buf_get_name(0)

    -- 3. Run the command 'tinymist.exportPdf'
    -- Args: [file_path, options_object]
    vim.lsp.buf.execute_command({
      command = "tinymist.exportPdf",
      arguments = {
        file_path, 
        {}
      }
    })
  end,
})
-- Setup lazy.nvim
require("lazy").setup({
  spec = {
    { import = "plugins" },
    -- add your plugins here
  },
  -- Configure any other settings here. See the documentation for more details.
  -- colorscheme that will be used when installing plugins.
  install = { colorscheme = { "everforest" } },
  -- automatically check for plugin updates
  checker = { enabled = true, notify = false },

})
