//! `handwired-sign`: make keys, sign program files, and check signatures. Keys are raw files: a secret key is 64
//! bytes (seed, then public key), a public key 32.

use std::io::{Read, Write};
use std::process::ExitCode;

use handwired_rs_sign::{PUBLIC_KEY_BYTES, SECRET_KEY_BYTES};

const USAGE: &str = "usage:\n\
    \x20 handwired-sign keygen SECRET_KEY PUBLIC_KEY\n\
    \x20 handwired-sign sign --key SECRET_KEY --slot SLOT --sequence N PROGRAM [SIGNATURE]\n\
    \x20 handwired-sign verify --key PUBLIC_KEY [--key PUBLIC_KEY ...] PROGRAM [SIGNATURE]\n\
    \x20 handwired-sign show SIGNATURE\n\
    SIGNATURE defaults to PROGRAM.sig. keygen reads its seed from /dev/urandom and won't overwrite a file.\n\
    verify exits 0 only if a trusted key signed PROGRAM, and prints its slot and sequence; the device must still\n\
    refuse a sequence below the last one it accepted for that slot.";

enum Failure {
    /// The command line is wrong.
    Usage,
    /// The command failed, for this reason.
    Because(String),
}

impl<E: std::fmt::Display> From<E> for Failure {
    fn from(error: E) -> Failure {
        Failure::Because(error.to_string())
    }
}

/// `error`, prefixed with the file it's about.
fn about(path: &str, error: impl std::fmt::Display) -> Failure {
    Failure::Because(format!("{path}: {error}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("keygen") => keygen(&args[1..]),
        Some("sign") => sign(&args[1..]),
        Some("verify") => verify(&args[1..]),
        Some("show") => show(&args[1..]),
        _ => Err(Failure::Usage),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage) => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::Because(message)) => {
            eprintln!("handwired-sign: {message}");
            ExitCode::FAILURE
        }
    }
}

fn read(path: &str) -> Result<Vec<u8>, Failure> {
    std::fs::read(path).map_err(|error| about(path, error))
}

fn read_key<const N: usize>(path: &str, what: &str) -> Result<[u8; N], Failure> {
    let bytes = read(path)?;
    bytes.as_slice().try_into().map_err(|_| about(path, format!("{} bytes; a {what} key is {N}", bytes.len())))
}

/// Writes a new file, refusing to replace one; `private` makes it readable by its owner only.
fn create(path: &str, bytes: &[u8], private: bool) -> Result<(), Failure> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    if private {
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    }
    #[cfg(not(unix))]
    let _ = private;
    options.open(path).and_then(|mut file| file.write_all(bytes)).map_err(|error| about(path, error))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// `--name value` pairs, and the positional arguments.
type Options<'a> = (Vec<(&'a str, &'a str)>, Vec<&'a str>);

/// Splits `--name value` options from positional arguments.
fn options(args: &[String]) -> Result<Options<'_>, Failure> {
    let (mut named, mut positional) = (Vec::new(), Vec::new());
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.strip_prefix("--") {
            Some(name) => named.push((name, args.next().ok_or(Failure::Usage)?.as_str())),
            None => positional.push(arg.as_str()),
        }
    }
    Ok((named, positional))
}

/// The value of an option given exactly once.
fn one<'a>(named: &[(&str, &'a str)], name: &str) -> Result<&'a str, Failure> {
    match named.iter().filter(|(key, _)| *key == name).collect::<Vec<_>>().as_slice() {
        [(_, value)] => Ok(value),
        _ => Err(Failure::Usage),
    }
}

/// PROGRAM [SIGNATURE], the signature defaulting to PROGRAM.sig.
fn program_and_signature(positional: &[&str]) -> Result<(String, String), Failure> {
    match positional {
        [program] => Ok((program.to_string(), format!("{program}.sig"))),
        [program, signature] => Ok((program.to_string(), signature.to_string())),
        _ => Err(Failure::Usage),
    }
}

fn keygen(args: &[String]) -> Result<(), Failure> {
    let [secret_path, public_path] = args else { return Err(Failure::Usage) };
    let mut seed = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut seed))
        .map_err(|error| about("/dev/urandom", error))?;
    let secret = handwired_rs_sign::secret_key_from_seed(seed)?;
    let public = handwired_rs_sign::public_key(&secret)?;
    create(secret_path, &secret, true)?;
    create(public_path, &public, false)?;
    println!("public key {}", hex(&public));
    Ok(())
}

fn sign(args: &[String]) -> Result<(), Failure> {
    let (named, positional) = options(args)?;
    if named.len() != 3 {
        return Err(Failure::Usage);
    }
    let secret = read_key::<SECRET_KEY_BYTES>(one(&named, "key")?, "secret")?;
    let slot = one(&named, "slot")?;
    let sequence = one(&named, "sequence")?.parse::<u64>().map_err(|_| Failure::Usage)?;
    let (program_path, signature_path) = program_and_signature(&positional)?;
    let program = read(&program_path)?;
    // Only what starts like a program, so a key isn't spent on the wrong file by accident. Its structure is
    // handwired-run's to check, which keeps this crate free of the runner.
    if !program.starts_with(b"HWCIRC\0\0") {
        return Err(about(&program_path, "not a handwired program"));
    }
    let signature = handwired_rs_sign::sign(&program, slot, sequence, &secret)?;
    std::fs::write(&signature_path, signature).map_err(|error| about(&signature_path, error))
}

fn verify(args: &[String]) -> Result<(), Failure> {
    let (named, positional) = options(args)?;
    if named.is_empty() || named.iter().any(|(name, _)| *name != "key") {
        return Err(Failure::Usage);
    }
    let trusted =
        named.iter().map(|(_, path)| read_key::<PUBLIC_KEY_BYTES>(path, "public")).collect::<Result<Vec<_>, _>>()?;
    let (program_path, signature_path) = program_and_signature(&positional)?;
    let (program, signature) = (read(&program_path)?, read(&signature_path)?);
    let signed =
        handwired_rs_sign::verify(&program, &signature, &trusted).map_err(|error| about(&signature_path, error))?;
    println!(
        "{program_path}: slot {}, sequence {}, signed by {}",
        signed.slot,
        signed.sequence,
        hex(&trusted[signed.key])
    );
    Ok(())
}

fn show(args: &[String]) -> Result<(), Failure> {
    let [path] = args else { return Err(Failure::Usage) };
    let bytes = read(path)?;
    let claim = handwired_rs_sign::read(&bytes).map_err(|error| about(path, error))?;
    println!("slot {}, sequence {}, key {} (not checked)", claim.slot, claim.sequence, hex(&claim.public_key));
    Ok(())
}
