//! This module implements the spartan SNARK protocol.
//! It provides the prover and verifier keys, as well as the SNARK itself.
use crate::{
  CommitmentKey,
  bellpepper::{
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
    zk_r1cs::{PrecommittedState, SpartanShape, SpartanWitness},
  },
  digest::{DigestComputer, SimpleDigestible},
  errors::SpartanError,
  math::Math,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
  },
  provider::{
    pcs::{
      hyrax_pc::HyraxPCS,
      ipa::{InnerProductArgumentLinear, InnerProductInstance, InnerProductWitness},
    },
    traits::{DlogGroup, DlogGroupExt},
  },
  r1cs::{R1CSWitness, SparseMatrix, SplitR1CSInstance, SplitR1CSShape},
  start_span,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::PCSEngineTrait,
    snark::{DigestHelperTrait, R1CSSNARKTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
  zk_sumcheck::SumcheckProof,
};
use ff::Field;
use group::prime::{PrimeCurve, PrimeCurveAffine};
use once_cell::sync::OnceCell;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info, info_span};

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  S: SplitR1CSShape<E>,
  vk_digest: SpartanDigest, // digest of the verifier's key
}

impl<E: Engine> SpartanProverKey<E> {
  /// Returns sizes associated with the SplitR1CSShape.
  /// It returns an array of 10 elements containing:
  /// [num_cons_unpadded, num_shared_unpadded, num_precommitted_unpadded, num_rest_unpadded,
  ///  num_cons, num_shared, num_precommitted, num_rest,
  ///  num_public, num_challenges]
  pub fn sizes(&self) -> [usize; 10] {
    self.S.sizes()
  }
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanVerifierKey<E: Engine> {
  vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey,
  S: SplitR1CSShape<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> SimpleDigestible for SpartanVerifierKey<E> {}

impl<E: Engine> DigestHelperTrait<E> for SpartanVerifierKey<E> {
  /// Returns the digest of the verifier's key.
  fn digest(&self) -> Result<SpartanDigest, SpartanError> {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<_>::new(self);
        dc.digest()
      })
      .cloned()
      .map_err(|_| SpartanError::DigestError {
        reason: "Unable to compute digest for SpartanVerifierKey".to_string(),
      })
  }
}

/// Binds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
pub(crate) fn compute_eval_table_sparse<E: Engine>(
  S: &SplitR1CSShape<E>,
  rx: &[E::Scalar],
) -> (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>) {
  assert_eq!(rx.len(), S.num_cons);

  let inner = |M: &SparseMatrix<E::Scalar>, M_evals: &mut Vec<E::Scalar>| {
    for (row_idx, ptrs) in M.indptr.windows(2).enumerate() {
      for (val, col_idx) in M.get_row_unchecked(ptrs.try_into().unwrap()) {
        M_evals[*col_idx] += rx[row_idx] * val;
      }
    }
  };

  let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
  let (A_evals, (B_evals, C_evals)) = rayon::join(
    || {
      let mut A_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
      inner(&S.A, &mut A_evals);
      A_evals
    },
    || {
      rayon::join(
        || {
          let mut B_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
          inner(&S.B, &mut B_evals);
          B_evals
        },
        || {
          let mut C_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
          inner(&S.C, &mut C_evals);
          C_evals
        },
      )
    },
  );

  (A_evals, B_evals, C_evals)
}

/// A type that holds the pre-processed state for proving
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PrepSNARK<E: Engine> {
  ps: PrecommittedState<E>,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSSNARK<E: Engine>
where
  E::GE: DlogGroupExt,
{
  U: SplitR1CSInstance<E>,
  sc_proof_outer: SumcheckProof<E>,
  ipa_proof_outer: InnerProductArgumentLinear<E>,
  ipa_instance_outer: InnerProductInstance<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),
  sc_proof_inner: SumcheckProof<E>,
  ipa_proof_inner: InnerProductArgumentLinear<E>,
  ipa_instance_inner: InnerProductInstance<E>,
  eval_W: E::Scalar,
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

impl<E: Engine<PCS = HyraxPCS<E>>> R1CSSNARKTrait<E> for R1CSSNARK<E>
where
  E::GE: DlogGroupExt,
  E::GE: PrimeCurve<Affine = <E::GE as DlogGroup>::AffineGroupElement, Scalar = E::Scalar>,
  <E::GE as PrimeCurve>::Affine: Send + Sync + PrimeCurveAffine<Scalar = E::Scalar, Curve = E::GE>,
{
  type ProverKey = SpartanProverKey<E>;
  type VerifierKey = SpartanVerifierKey<E>;
  type PrepSNARK = PrepSNARK<E>;

  fn setup<C: SpartanCircuit<E>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    let S = ShapeCS::r1cs_shape(&circuit)?;
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S])?;

    let vk = SpartanVerifierKey {
      S: S.clone(),
      vk_ee,
      digest: OnceCell::new(),
    };
    let pk = Self::ProverKey {
      ck,
      S,
      vk_digest: vk.digest()?,
    };

    Ok((pk, vk))
  }

  /// Prepares the SNARK for proving
  fn prep_prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<Self::PrepSNARK, SpartanError> {
    let mut ps = SatisfyingAssignment::shared_witness(&pk.S, &pk.ck, &circuit, is_small)?;
    SatisfyingAssignment::precommitted_witness(&mut ps, &pk.S, &pk.ck, &circuit, is_small)?;

    Ok(PrepSNARK { ps })
  }

  /// produces a succinct proof of satisfiability of an R1CS instance
  fn prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    prep_snark: &Self::PrepSNARK,
    is_small: bool,
  ) -> Result<Self, SpartanError> {
    let mut prep_snark = prep_snark.clone(); // make a copy so we can modify it

    let mut transcript = <E as Engine>::TE::new(b"R1CSSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);

    let public_values = circuit
      .public_values()
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Circuit does not provide public IO: {e}"),
      })?;

    // absorb the public values into the transcript
    transcript.absorb(b"public_values", &public_values.as_slice());

    let (U, W) = SatisfyingAssignment::r1cs_instance_and_witness(
      &mut prep_snark.ps,
      &pk.S,
      &pk.ck,
      &circuit,
      is_small,
      &mut transcript,
    )?;

    Self::prove_inner(pk, U, W, &mut transcript)
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError> {
    let (_verify_span, verify_t) = start_span!("r1cs_snark_verify");
    let mut transcript = E::TE::new(b"R1CSSNARK");

    // append the digest of R1CS matrices
    transcript.absorb(b"vk", &vk.digest()?);

    // validate the provided split R1CS instance and convert to regular instance
    self.U.validate(&vk.S, &mut transcript)?;
    let U_regular = self.U.to_regular_instance()?;

    let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    info!(
      "Verifying R1CS SNARK with {} rounds for outer sum-check and {} rounds for inner sum-check",
      num_rounds_x, num_rounds_y
    );

    // outer sum-check
    let (_tau_span, tau_t) = start_span!("compute_tau_verify");
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;
    info!(elapsed_ms = %tau_t.elapsed().as_millis(), "compute_tau_verify");

    transcript.absorb(
      b"outer sumcheck masks",
      &self.ipa_instance_outer.comm_a_vec(),
    );

    let (_outer_sumcheck_span, outer_sumcheck_t) = start_span!("outer_sumcheck_verify");
    let (claim_outer_final, r_x, _outer_lc_poly) =
      self
        .sc_proof_outer
        .verify(E::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;

    self.ipa_proof_outer.verify(
      &vk.vk_ee.ck()[..num_rounds_x * 3],
      &vk.vk_ee.h(),
      &vk.vk_ee.ck_s(),
      num_rounds_x * 3,
      &self.ipa_instance_outer,
      &mut transcript,
    )?;

    // verify claim_outer_final
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let taus_bound_rx = tau.evaluate(&r_x);

    let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
    if claim_outer_final_expected != claim_outer_final - self.ipa_instance_outer.c() {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %outer_sumcheck_t.elapsed().as_millis(), "outer_sumcheck_verify");

    transcript.absorb(
      b"claims_outer",
      &[
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2,
      ]
      .as_slice(),
    );

    // inner sum-check
    let (_inner_sumcheck_span, inner_sumcheck_t) = start_span!("inner_sumcheck_verify");
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint =
      self.claims_outer.0 + r * self.claims_outer.1 + r * r * self.claims_outer.2;

    transcript.absorb(
      b"inner sumcheck masks",
      &self.ipa_instance_inner.comm_a_vec(),
    );

    let (claim_inner_final, r_y, _inner_lc_poly) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)?;

    self.ipa_proof_inner.verify(
      &vk.vk_ee.ck()[..num_rounds_y * 2],
      &vk.vk_ee.h(),
      &vk.vk_ee.ck_s(),
      num_rounds_y * 2,
      &self.ipa_instance_inner,
      &mut transcript,
    )?;

    // verify claim_inner_final
    let eval_Z = {
      let eval_X = {
        // public IO is (1, X)
        let X = vec![E::Scalar::ONE]
          .into_iter()
          .chain(U_regular.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(num_vars.log_2(), X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    // compute evaluations of R1CS matrices
    let (_matrix_eval_span, matrix_eval_t) = start_span!("matrix_evaluations");
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
          M.indptr
            .par_windows(2)
            .enumerate()
            .map(|(row_idx, ptrs)| {
              M.get_row_unchecked(ptrs.try_into().unwrap())
                .map(|(val, col_idx)| {
                  let prod = T_x[row_idx] * T_y[*col_idx];
                  if *val == E::Scalar::ONE {
                    prod
                  } else if *val == -E::Scalar::ONE {
                    -prod
                  } else {
                    prod * val
                  }
                })
                .sum::<E::Scalar>()
            })
            .sum()
        };

      let (T_x, T_y) = rayon::join(
        || EqPolynomial::evals_from_points(r_x),
        || EqPolynomial::evals_from_points(r_y),
      );

      (0..M_vec.len())
        .into_par_iter()
        .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
        .collect()
    };

    let evals = multi_evaluate(&[&vk.S.A, &vk.S.B, &vk.S.C], &r_x, &r_y);

    let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;
    if claim_inner_final - self.ipa_instance_inner.c() != claim_inner_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    info!(elapsed_ms = %matrix_eval_t.elapsed().as_millis(), "matrix_evaluations");
    info!(elapsed_ms = %inner_sumcheck_t.elapsed().as_millis(), "inner_sumcheck_verify");

    // verify
    let (_pcs_verify_span, pcs_verify_t) = start_span!("pcs_verify");
    <E as Engine>::PCS::verify(
      &vk.vk_ee,
      &mut transcript,
      &U_regular.comm_W,
      &r_y[1..],
      &self.eval_W,
      &self.eval_arg,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pcs_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "r1cs_snark_verify");
    Ok(self.U.public_values.clone())
  }
}

impl<E: Engine<PCS = HyraxPCS<E>>> R1CSSNARK<E>
where
  E::GE: DlogGroupExt,
  E::GE: PrimeCurve<Affine = <E::GE as DlogGroup>::AffineGroupElement, Scalar = E::Scalar>,
  <E::GE as PrimeCurve>::Affine: Send + Sync + PrimeCurveAffine<Scalar = E::Scalar, Curve = E::GE>,
{
  ///
  pub fn prove_inner(
    pk: &<Self as R1CSSNARKTrait<E>>::ProverKey,
    U: SplitR1CSInstance<E>,
    W: R1CSWitness<E>,
    transcript: &mut <E as Engine>::TE,
  ) -> Result<Self, SpartanError> {
    let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(pk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    let mut outer_sumcheck_masks = vec![];
    for _ in 0..num_rounds_x {
      let mask = [
        E::Scalar::random(&mut OsRng),
        E::Scalar::random(&mut OsRng),
        E::Scalar::random(&mut OsRng),
      ];
      outer_sumcheck_masks.push(mask);
    }

    let mut inner_sumcheck_masks = vec![];
    for _ in 0..num_rounds_y {
      let mask = [E::Scalar::random(&mut OsRng), E::Scalar::random(&mut OsRng)];
      inner_sumcheck_masks.push(mask);
    }

    // compute the full satisfying assignment by concatenating W.W, 1, and U.X
    let mut z = [
      W.W.clone(),
      vec![E::Scalar::ONE],
      U.public_values.clone(),
      U.challenges.clone(),
    ]
    .concat();

    let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(pk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check preparation
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;

    let (_poly_tau_span, poly_tau_t) = start_span!("prepare_poly_tau");
    let mut poly_tau = MultilinearPolynomial::new(tau.evals());
    info!(elapsed_ms = %poly_tau_t.elapsed().as_millis(), "prepare_poly_tau");

    let (_mv_span, mv_t) = start_span!("matrix_vector_multiply");
    let (Az, Bz, Cz) = pk.S.multiply_vec(&z)?;
    info!(
      elapsed_ms = %mv_t.elapsed().as_millis(),
      constraints = %pk.S.num_cons,
      vars = %num_vars,
      "matrix_vector_multiply"
    );

    let (_mp_span, mp_t) = start_span!("prepare_multilinear_polys");
    let (mut poly_Az, mut poly_Bz, mut poly_Cz) = (
      MultilinearPolynomial::new(Az),
      MultilinearPolynomial::new(Bz),
      MultilinearPolynomial::new(Cz),
    );
    info!(elapsed_ms = %mp_t.elapsed().as_millis(), "prepare_multilinear_polys");

    // outer sum-check
    let (_sc_span, sc_t) = start_span!("outer_sumcheck");

    let comb_func_outer =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    let masks = outer_sumcheck_masks
      .iter()
      .flatten()
      .copied()
      .collect::<Vec<_>>();

    let comm_masks =
      <E as Engine>::GE::vartime_multiscalar_mul(&masks, &pk.ck.ck()[..masks.len()], true)?;
    let r_a = E::Scalar::random(&mut OsRng);
    let comm_masks = comm_masks + pk.ck.h() * r_a;

    transcript.absorb(b"outer sumcheck masks", &comm_masks);

    let (sc_proof_outer, r_x, claims_outer, outer_lc_poly) =
      SumcheckProof::prove_cubic_with_additive_term(
        &E::Scalar::ZERO, // claim is zero
        num_rounds_x,
        &outer_sumcheck_masks,
        &mut poly_tau,
        &mut poly_Az,
        &mut poly_Bz,
        &mut poly_Cz,
        comb_func_outer,
        transcript,
      )?;

    let (ipa_proof_outer, ipa_instance_outer) = {
      let mask_eval = masks
        .iter()
        .zip(outer_lc_poly.iter())
        .map(|(mask, lc)| *mask * lc)
        .sum();

      let ipa_instance = InnerProductInstance::<E>::new(&comm_masks, &outer_lc_poly, &mask_eval);

      let ipa_witness = InnerProductWitness::<E>::new(&masks, &r_a);

      let ipa_proof = InnerProductArgumentLinear::<E>::prove(
        &pk.ck.ck()[..masks.len()],
        &pk.ck.h(),
        &pk.ck.ck_s(),
        &ipa_instance,
        &ipa_witness,
        transcript,
      )?;

      (ipa_proof, ipa_instance)
    };

    // claims from the end of sum-check
    let (claim_Az, claim_Bz, claim_Cz): (E::Scalar, E::Scalar, E::Scalar) =
      (claims_outer[1], claims_outer[2], claims_outer[3]);
    transcript.absorb(b"claims_outer", &[claim_Az, claim_Bz, claim_Cz].as_slice());
    info!(elapsed_ms = %sc_t.elapsed().as_millis(), "outer_sumcheck");

    // inner sum-check preparation
    let (_r_span, r_t) = start_span!("prepare_inner_claims");
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;
    info!(elapsed_ms = %r_t.elapsed().as_millis(), "prepare_inner_claims");

    let (_eval_rx_span, eval_rx_t) = start_span!("compute_eval_rx");
    let evals_rx = EqPolynomial::evals_from_points(&r_x.clone());
    info!(elapsed_ms = %eval_rx_t.elapsed().as_millis(), "compute_eval_rx");

    let (_sparse_span, sparse_t) = start_span!("compute_eval_table_sparse");
    let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&pk.S, &evals_rx);
    info!(elapsed_ms = %sparse_t.elapsed().as_millis(), "compute_eval_table_sparse");

    let (_abc_span, abc_t) = start_span!("prepare_poly_ABC");
    assert_eq!(evals_A.len(), evals_B.len());
    assert_eq!(evals_A.len(), evals_C.len());
    let poly_ABC = (0..evals_A.len())
      .into_par_iter()
      .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
      .collect::<Vec<E::Scalar>>();
    info!(elapsed_ms = %abc_t.elapsed().as_millis(), "prepare_poly_ABC");

    let (_z_span, z_t) = start_span!("prepare_poly_z");
    let poly_z = {
      z.resize(num_vars * 2, E::Scalar::ZERO);
      z
    };
    info!(elapsed_ms = %z_t.elapsed().as_millis(), "prepare_poly_z");

    // inner sum-check
    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck");

    debug!("Proving inner sum-check with {} rounds", num_rounds_y);
    debug!(
      "Inner sum-check sizes - poly_ABC: {}, poly_z: {}",
      poly_ABC.len(),
      poly_z.len()
    );
    let comb_func = |poly_A_comp: &E::Scalar, poly_B_comp: &E::Scalar| -> E::Scalar {
      *poly_A_comp * *poly_B_comp
    };

    let masks = inner_sumcheck_masks
      .iter()
      .flatten()
      .copied()
      .collect::<Vec<_>>();

    let comm_masks =
      <E as Engine>::GE::vartime_multiscalar_mul(&masks, &pk.ck.ck()[..masks.len()], true)?;
    let r_a = E::Scalar::random(&mut OsRng);
    let comm_masks = comm_masks + pk.ck.h() * r_a;

    transcript.absorb(b"inner sumcheck masks", &comm_masks);

    let (sc_proof_inner, r_y, _claims_inner, inner_lc_poly) = SumcheckProof::prove_quad(
      &claim_inner_joint,
      num_rounds_y,
      &inner_sumcheck_masks,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      comb_func,
      transcript,
    )?;
    info!(elapsed_ms = %sc2_t.elapsed().as_millis(), "inner_sumcheck");

    let (ipa_proof_inner, ipa_instance_inner) = {
      let mask_eval = masks
        .iter()
        .zip(inner_lc_poly.iter())
        .map(|(mask, lc)| *mask * lc)
        .sum();

      let ipa_instance = InnerProductInstance::<E>::new(&comm_masks, &inner_lc_poly, &mask_eval);

      let ipa_witness = InnerProductWitness::<E>::new(&masks, &r_a);

      let ipa_proof = InnerProductArgumentLinear::<E>::prove(
        &pk.ck.ck()[..masks.len()],
        &pk.ck.h(),
        &pk.ck.ck_s(),
        &ipa_instance,
        &ipa_witness,
        transcript,
      )?;

      (ipa_proof, ipa_instance)
    };

    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let U_regular = U.to_regular_instance()?;
    let (eval_W, eval_arg) = E::PCS::prove(
      &pk.ck,
      transcript,
      &U_regular.comm_W,
      &W.W,
      &W.r_W,
      &r_y[1..],
    )?;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    Ok(R1CSSNARK {
      U,
      sc_proof_outer,
      ipa_proof_outer,
      ipa_instance_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      sc_proof_inner,
      ipa_proof_inner,
      ipa_instance_inner,
      eval_W,
      eval_arg,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit {}

  impl<E: Engine> SpartanCircuit<E> for CubicCircuit {
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::from(15u64)])
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      // In this example, we do not have shared variables.
      Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<<E as Engine>::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>], // shared variables, if any
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
      // In this example, we do not have precommitted variables.
      Ok(vec![])
    }

    fn num_challenges(&self) -> usize {
      // In this example, we do not use challenges.
      0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
      _: &[AllocatedNum<E::Scalar>],
      _: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ONE + E::Scalar::ONE))?;
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + E::Scalar::from(5u64))
      })?;

      cs.enforce(
        || "y = x^3 + x + 5",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      let _ = y.inputize(cs.namespace(|| "output"));

      Ok(())
    }
  }

  #[test]
  fn test_snark() {
    type E = crate::provider::PallasHyraxEngine;
    // type S = R1CSSNARK<E>;
    test_snark_with::<E>();

    type E2 = crate::provider::T256HyraxEngine;
    // type S2 = R1CSSNARK<E2>;
    test_snark_with::<E2>();
  }

  fn test_snark_with<E: Engine<PCS = HyraxPCS<E>>>()
  where
    E::GE: DlogGroupExt,
    E::GE: PrimeCurve<Affine = <E::GE as DlogGroup>::AffineGroupElement, Scalar = E::Scalar>,
    <E::GE as PrimeCurve>::Affine:
      Send + Sync + PrimeCurveAffine<Scalar = E::Scalar, Curve = E::GE>,
  {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) = R1CSSNARK::<E>::setup(circuit.clone()).unwrap();

    // generate pre-processed state for proving
    let mut prep_snark = R1CSSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();

    let mut transcript = <E as Engine>::TE::new(b"R1CSSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);

    let public_values = SpartanCircuit::<E>::public_values(&circuit)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Circuit does not provide public IO: {e}"),
      })
      .unwrap();

    // absorb the public values into the transcript
    transcript.absorb(b"public_values", &public_values.as_slice());

    let (U, W) = SatisfyingAssignment::r1cs_instance_and_witness(
      &mut prep_snark.ps,
      &pk.S,
      &pk.ck,
      &circuit,
      false,
      &mut transcript,
    )
    .unwrap();

    // generate a witness and proof
    let res = R1CSSNARK::<E>::prove_inner(&pk, U.clone(), W.clone(), &mut transcript);
    // assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk);
    // assert!(res.is_ok());
    assert_eq!(res.unwrap(), [<E as Engine>::Scalar::from(15u64)]);

    // Reblind instance and witness
    let mut reblind_transcript = <E as Engine>::TE::new(b"R1CSSNARK");
    reblind_transcript.absorb(b"vk", &pk.vk_digest);

    let public_values = SpartanCircuit::<E>::public_values(&circuit)
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Circuit does not provide public IO: {e}"),
      })
      .unwrap();

    // absorb the public values into the reblind_transcript
    reblind_transcript.absorb(b"public_values", &public_values.as_slice());

    let (U, W) = SatisfyingAssignment::reblind_r1cs_instance_and_witness(
      U,
      W,
      &pk.ck,
      &mut reblind_transcript,
    )
    .unwrap();

    // generate a witness and proof
    let res = R1CSSNARK::<E>::prove_inner(&pk, U, W, &mut reblind_transcript);
    // assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk);
    // assert!(res.is_ok());
    assert_eq!(res.unwrap(), [<E as Engine>::Scalar::from(15u64)])
  }
}
