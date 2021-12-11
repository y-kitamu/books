;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((cpp-mode . ((lsp-docker+-server-id . ccls)
               (lsp-docker+-docker-server-id . ccls-docker)
               (lsp-docker+-server-command . "ccls")
               (lsp-docker+-path-mappings . (("${HOME}/work" . "${HOME}/work")))
               (lsp-docker+-docker-options . "-u ${USER}")
               (lsp-docker+-image-id . "cpp_engine")
               (lsp-docker+-container-name . "cpp_engine")
               (lsp-docker+-server-cmd-fn . lsp-docker+-exec-in-container))))
