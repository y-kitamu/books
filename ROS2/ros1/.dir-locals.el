;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((c++-mode
  . ((lsp-docker+-server-id . ccls)
     (lsp-docker+-docker-server-id . ccls-docker)
     (lsp-docker+-server-command . "ccls")
     (lsp-docker+-image-id . "docker_ros1")
     (lsp-docker+-container-name . "docker_ros1_1")
     (lsp-docker+-server-cmd-fn . lsp-docker+-exec-in-container)
     (lsp-docker+-docker-options . "--user ${USER}")
     (lsp-docker+-path-mappings . (("${HOME}/work/Learning/Book/ROS2/" . "${HOME}/work/"))))))
