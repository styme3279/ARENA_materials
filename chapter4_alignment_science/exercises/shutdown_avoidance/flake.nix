{
  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/nixpkgs-unstable;
    flake-utils.url = github:numtide/flake-utils;
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
    with nixpkgs.legacyPackages.${system};
    let
      scripts = {
        "run" = ''
          set -e
          pyright .
          inspect eval shutdown.py "$@"
        '';
        "next_question" = "gen && go run next_question.go strings.go";
      };
    in {
      devShell = stdenvNoCC.mkDerivation {
        name = "shell";
        buildInputs = lib.attrsets.mapAttrsToList writeShellScriptBin scripts ++ [
          go
          uv
        ];
        shellHook = ''
          export INSPECT_EVAL_REASONING_SUMMARY=auto;
        '';
      };
    });
}
