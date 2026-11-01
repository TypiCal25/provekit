mod whir_r1cs;

use {
    crate::whir_r1cs::WhirR1CSVerifier,
    anyhow::{ensure, Context, Result},
    provekit_common::{NoirProof, Verifier, WhirR1CSScheme, R1CS},
    tracing::instrument,
};

pub trait Verify {
    fn verify(&mut self, proof: &NoirProof) -> Result<()>;
}

impl Verify for Verifier {
    #[instrument(skip_all)]
    fn verify(&mut self, proof: &NoirProof) -> Result<()> {
        provekit_common::register_ntt();

        self.whir_for_witness
            .take()
            .context("Verifier has already been consumed; cannot verify twice")?
            .verify(&proof.whir_r1cs_proof, &proof.public_inputs, &self.r1cs)?;

        Ok(())
    }
}

/// A verifier that checks its R1CS against the WHIR scheme's R1CS hash once,
/// when it is built, and can then verify any number of proofs.
///
/// [`Verify::verify`] re-serialises and hashes the whole R1CS for every proof
/// and can be used only once. `CheckedVerifier` makes the same hash check in
/// [`CheckedVerifier::new`] and runs every other check per proof, so it
/// accepts and rejects exactly the proofs a fresh [`Verifier`] would. The R1CS
/// and scheme are private and never mutated, so each proof is checked against
/// the pair whose hashes matched.
#[derive(Debug)]
pub struct CheckedVerifier {
    r1cs:             R1CS,
    whir_for_witness: WhirR1CSScheme,
}

impl CheckedVerifier {
    /// Builds a reusable verifier from `verifier`, failing if its R1CS does not
    /// hash to the scheme's R1CS hash or it has already been consumed.
    #[instrument(skip_all)]
    pub fn new(verifier: Verifier) -> Result<Self> {
        let whir_for_witness = verifier
            .whir_for_witness
            .context("Verifier has already been consumed; cannot build a checked verifier")?;
        ensure!(
            verifier.r1cs.hash() == whir_for_witness.r1cs_hash,
            "R1CS hash mismatch"
        );
        Ok(Self {
            r1cs: verifier.r1cs,
            whir_for_witness,
        })
    }

    /// Verifies `proof` with every check [`Verify::verify`] makes; the R1CS
    /// hash check was made once in [`CheckedVerifier::new`].
    #[instrument(skip_all)]
    pub fn verify(&self, proof: &NoirProof) -> Result<()> {
        provekit_common::register_ntt();

        self.whir_for_witness.verify_with_checked_r1cs(
            &proof.whir_r1cs_proof,
            &proof.public_inputs,
            &self.r1cs,
        )
    }
}
