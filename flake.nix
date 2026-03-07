{
  description = "Development flake for Boolmesh";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url  = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
    gimli-rs = {
      url = "github:gimli-rs/addr2line";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    fenix,
    crane,
    gimli-rs,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        fenix-pkgs = fenix.packages.${system};
        fenix-channel = fenix-pkgs.stable;
        fenix-toolchain = fenix-channel.toolchain;
        craneLib = (crane.mkLib pkgs).overrideScope (final: prev: {
          cargo = fenix-channel.cargo;
          rustc = fenix-channel.rustc;
        });

        addr2line = craneLib.buildPackage {
          src = craneLib.cleanCargoSource gimli-rs;
	  strictDeps = true;

          cargoExtraArgs = "--features=bin";

          # Tests are broken due to the test fixture not being present in the repo:
          # https://github.com/gimli-rs/addr2line/blob/fbd5e23630fafe5b69d7ddd4d744bfd550f28655/tests/parse.rs#L9
          doCheck = false;
        };
      in rec
      {
        devShells.default = with pkgs; pkgs.mkShell {
          buildInputs = [
	    bashInteractive
            fenix-pkgs.rust-analyzer
            fenix-toolchain
            cargo-expand
            cargo-flamegraph

            # For Bevy
            pkg-config
            alsa-lib
            udev
            wayland
            libxkbcommon
            vulkan-loader
          ];
  
          LD_LIBRARY_PATH = with pkgs; pkgs.lib.makeLibraryPath [
            libxkbcommon
            vulkan-loader
            udev
            wayland
            alsa-lib
          ];

	  shellHook = ''
            export SHELL=${pkgs.bashInteractive}/bin/bash

            # There are *so many* versions of addr2line provided by other packages.
            # We have to do this slight hack to make sure this version takes priority.
            export PATH=${addr2line}/bin:$PATH
          '';
        };

      }
    );
}
