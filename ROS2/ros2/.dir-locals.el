;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((c++-mode
  . ((lsp-docker+-server-id . ccls)
     (lsp-docker+-docker-server-id . ccls-docker)
     (lsp-docker+-server-command . "ccls")
     (lsp-docker+-image-id . "docker_ros2")
     (lsp-docker+-container-name . "docker_ros2_1")
     (lsp-docker+-server-cmd-fn . lsp-docker+-exec-in-container)
     (lsp-docker+-docker-options . "--user ${USER}")
     (lsp-docker+-path-mappings . (("${HOME}/work/Learning/Book/ROS2/" . "${HOME}/work/")))))
 (python-mode
  . ((lsp-docker+-server-id . pyright)
     (lsp-docker+-docker-server-id . pyls-docker)
     (lsp-docker+-server-command . "pyright-langserver --stdio")
     (lsp-docker+-image-id . "docker_ros2")
     (lsp-docker+-container-name . "docker_ros2_pyright")
     (lsp-docker+-server-cmd-fn . lsp-docker+-exec-in-container)
     (lsp-docker+-docker-options . "--user ${USER}")
     (lsp-docker+-path-mappings . (("${HOME}/work/Learning/Book/ROS2/" . "${HOME}/work/"))))))
