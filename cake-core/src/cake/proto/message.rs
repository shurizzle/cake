use std::borrow::Cow;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use speedy::{BigEndian, Readable, Writable};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const _: () = {
    use candle_core::DType;

    const fn _check_dtype_enum(dtype: DType) -> usize {
        match dtype {
            DType::U8
            | DType::U32
            | DType::I64
            | DType::BF16
            | DType::F16
            | DType::F32
            | DType::F64 => 0,
        }
    }

    const fn check_dtype_enum() -> usize {
        _check_dtype_enum(DType::U8)
    }

    const fn layout_eq<T1, T2>() -> usize {
        if core::mem::size_of::<T1>() == core::mem::size_of::<T2>()
            && core::mem::align_of::<T1>() == core::mem::align_of::<T2>()
        {
            0
        } else {
            1
        }
    }

    const _: [(); 0] = [(); check_dtype_enum()];
    const _: [(); 0] = [(); layout_eq::<DType, u8>()];
};

mod consts {
    use candle_core::DType;

    pub const U8: u8 = DType::U8 as u8;
    pub const U32: u8 = DType::U32 as u8;
    pub const I64: u8 = DType::I64 as u8;
    pub const BF16: u8 = DType::BF16 as u8;
    pub const F16: u8 = DType::F16 as u8;
    pub const F32: u8 = DType::F32 as u8;
    pub const F64: u8 = DType::F64 as u8;
}

fn read_dtype<'de, C: speedy::Context, R: speedy::Reader<'de, C>>(
    reader: &mut R,
) -> core::result::Result<DType, <C as speedy::Context>::Error> {
    match reader.read_u8()? {
        consts::U8 => Ok(DType::U8),
        consts::U32 => Ok(DType::U32),
        consts::I64 => Ok(DType::I64),
        consts::BF16 => Ok(DType::BF16),
        consts::F16 => Ok(DType::F16),
        consts::F32 => Ok(DType::F32),
        consts::F64 => Ok(DType::F64),
        _ => Err(speedy::private::error_invalid_enum_variant()),
    }
}

fn write_dtype<C: speedy::Context, W: ?Sized + speedy::Writer<C>>(
    value: DType,
    writer: &mut W,
) -> core::result::Result<(), <C as speedy::Context>::Error> {
    (value as u8).write_to(writer)
}

/// Represents a tensor in Cake protocol.
#[repr(C, u8)]
#[derive(Debug, Clone, PartialEq)]
pub enum RawTensor<'a> {
    U8(Cow<'a, [u8]>) = consts::U8,
    U32(Cow<'a, [u32]>) = consts::U32,
    I64(Cow<'a, [i64]>) = consts::I64,
    BF16(Cow<'a, [half::bf16]>) = consts::BF16,
    F16(Cow<'a, [half::f16]>) = consts::F16,
    F32(Cow<'a, [f32]>) = consts::F32,
    F64(Cow<'a, [f64]>) = consts::F64,
}

impl RawTensor<'_> {
    pub fn into_owned(self) -> RawTensor<'static> {
        match self {
            RawTensor::U8(s) => RawTensor::U8(Cow::Owned(s.into_owned())),
            RawTensor::U32(s) => RawTensor::U32(Cow::Owned(s.into_owned())),
            RawTensor::I64(s) => RawTensor::I64(Cow::Owned(s.into_owned())),
            RawTensor::BF16(s) => RawTensor::BF16(Cow::Owned(s.into_owned())),
            RawTensor::F16(s) => RawTensor::F16(Cow::Owned(s.into_owned())),
            RawTensor::F32(s) => RawTensor::F32(Cow::Owned(s.into_owned())),
            RawTensor::F64(s) => RawTensor::F64(Cow::Owned(s.into_owned())),
        }
    }

    pub fn dtype(&self) -> DType {
        unsafe { core::mem::transmute::<u8, DType>(*(self as *const Self as *const u8)) }
    }

    pub fn into_tensor(self, device: &Device) -> candle_core::Result<Tensor> {
        #[inline(always)]
        pub fn from_vec<D: candle_core::WithDType>(
            data: Vec<D>,
            device: &Device,
        ) -> candle_core::Result<Tensor> {
            let shape = data.len();
            Tensor::from_vec(data, shape, device)
        }

        match self {
            RawTensor::U8(data) => from_vec(data.into_owned(), device),
            RawTensor::U32(data) => from_vec(data.into_owned(), device),
            RawTensor::I64(data) => from_vec(data.into_owned(), device),
            RawTensor::BF16(data) => from_vec(data.into_owned(), device),
            RawTensor::F16(data) => from_vec(data.into_owned(), device),
            RawTensor::F32(data) => from_vec(data.into_owned(), device),
            RawTensor::F64(data) => from_vec(data.into_owned(), device),
        }
    }
}

impl<'de, C: speedy::Context> Readable<'de, C> for RawTensor<'de> {
    fn read_from<R: speedy::Reader<'de, C>>(
        reader: &mut R,
    ) -> core::result::Result<Self, <C as speedy::Context>::Error> {
        macro_rules! read_half {
            ($reader:expr $(,)?) => {{
                use ::half::{slice::HalfBitsSliceExt as HB, vec::HalfBitsVecExt as HV};
                use ::std::borrow::Cow;
                <Cow<[u16]> as ::speedy::Readable<_>>::read_from(reader).map(|v| match v {
                    Cow::Borrowed(b) => Cow::Borrowed(HB::reinterpret_cast(b)),
                    Cow::Owned(o) => Cow::Owned(HV::reinterpret_into(o)),
                })
            }};
        }

        match read_dtype(reader)? {
            DType::U8 => Cow::read_from(reader).map(RawTensor::U8),
            DType::U32 => Cow::read_from(reader).map(RawTensor::U32),
            DType::I64 => Cow::read_from(reader).map(RawTensor::I64),
            DType::BF16 => read_half!(reader).map(RawTensor::BF16),
            DType::F16 => read_half!(reader).map(RawTensor::F16),
            DType::F32 => Cow::read_from(reader).map(RawTensor::F32),
            DType::F64 => Cow::read_from(reader).map(RawTensor::F64),
        }
    }
}

impl<C: speedy::Context> Writable<C> for RawTensor<'_> {
    fn write_to<T: ?Sized + speedy::Writer<C>>(
        &self,
        writer: &mut T,
    ) -> core::result::Result<(), <C as speedy::Context>::Error> {
        use half::slice::HalfFloatSliceExt;

        write_dtype(self.dtype(), writer)?;

        match self {
            RawTensor::U8(v) => Writable::write_to(v, writer),
            RawTensor::U32(v) => Writable::write_to(v, writer),
            RawTensor::I64(v) => Writable::write_to(v, writer),
            RawTensor::BF16(v) => Writable::write_to(v.reinterpret_cast(), writer),
            RawTensor::F16(v) => Writable::write_to(v.reinterpret_cast(), writer),
            RawTensor::F32(v) => Writable::write_to(v, writer),
            RawTensor::F64(v) => Writable::write_to(v, writer),
        }
    }
}

impl TryFrom<&Tensor> for RawTensor<'static> {
    type Error = candle_core::Error;

    fn try_from(value: &Tensor) -> candle_core::Result<Self> {
        let vs = value.flatten_all()?;
        Ok(match vs.dtype() {
            DType::U8 => RawTensor::U8(Cow::Owned(vs.try_into()?)),
            DType::U32 => RawTensor::U32(Cow::Owned(vs.try_into()?)),
            DType::I64 => RawTensor::I64(Cow::Owned(vs.try_into()?)),
            DType::BF16 => RawTensor::BF16(Cow::Owned(vs.try_into()?)),
            DType::F16 => RawTensor::F16(Cow::Owned(vs.try_into()?)),
            DType::F32 => RawTensor::F32(Cow::Owned(vs.try_into()?)),
            DType::F64 => RawTensor::F64(Cow::Owned(vs.try_into()?)),
        })
    }
}

impl TryFrom<Tensor> for RawTensor<'static> {
    type Error = candle_core::Error;

    fn try_from(value: Tensor) -> candle_core::Result<Self> {
        RawTensor::try_from(&value)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Readable, Writable)]
#[speedy(tag_type = u8)]
pub enum DeviceType {
    #[default]
    Cpu,
    Cuda,
    Metal,
}

impl DeviceType {
    pub fn to_str(self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Cuda => "cuda",
            DeviceType::Metal => "metal",
        }
    }
}

impl core::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.to_str(), f)
    }
}

/// Diagnostic information about a worker.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkerInfo<'a> {
    /// Protocol version.
    pub version: Cow<'a, str>,
    /// Tensors data type.
    pub dtype: DType,
    /// Operating system.
    pub os: Cow<'a, str>,
    /// Architecture.
    pub arch: Cow<'a, str>,
    /// Device.
    pub device: DeviceType,
    /// Device index for multi GPU environments.
    pub device_idx: usize,
    /// Latency in millisenconds.
    pub latency: u128,
}

impl<'de, C: speedy::Context> Readable<'de, C> for WorkerInfo<'de> {
    fn read_from<R: speedy::Reader<'de, C>>(
        reader: &mut R,
    ) -> core::result::Result<Self, <C as speedy::Context>::Error> {
        Ok(WorkerInfo {
            version: Readable::read_from(reader)?,
            dtype: read_dtype(reader)?,
            os: Readable::read_from(reader)?,
            arch: Readable::read_from(reader)?,
            device: Readable::read_from(reader)?,
            device_idx: Readable::read_from(reader)?,
            latency: Readable::read_from(reader)?,
        })
    }
}

impl<C: speedy::Context> Writable<C> for WorkerInfo<'_> {
    fn write_to<T: ?Sized + speedy::Writer<C>>(
        &self,
        writer: &mut T,
    ) -> core::result::Result<(), <C as speedy::Context>::Error> {
        self.version.write_to(writer)?;
        write_dtype(self.dtype, writer)?;
        self.os.write_to(writer)?;
        self.arch.write_to(writer)?;
        self.device.write_to(writer)?;
        self.device_idx.write_to(writer)?;
        self.latency.write_to(writer)
    }
}

impl<'a> Default for WorkerInfo<'a> {
    fn default() -> Self {
        Self {
            version: Default::default(),
            dtype: DType::F16,
            os: Default::default(),
            arch: Default::default(),
            device: Default::default(),
            device_idx: Default::default(),
            latency: Default::default(),
        }
    }
}

impl<'a> WorkerInfo<'a> {
    pub fn into_owned(self) -> WorkerInfo<'static> {
        WorkerInfo {
            version: Cow::Owned(self.version.into_owned()),
            dtype: self.dtype,
            os: Cow::Owned(self.os.into_owned()),
            arch: Cow::Owned(self.arch.into_owned()),
            device: self.device,
            device_idx: self.device_idx,
            latency: self.latency,
        }
    }
}

/// A Cake protocol message.
#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Readable, Writable)]
#[speedy(tag_type = u8)]
pub enum Message<'a> {
    /// First message sent.
    Hello,
    /// Message that the worker sends when a master connects with runtime information.
    WorkerInfo(WorkerInfo<'a>),
    /// Single inference operation for a given layer.
    SingleOp {
        layer_name: Cow<'a, str>,
        x: RawTensor<'a>,
        index_pos: usize,
        block_idx: usize,
    },
    /// Batched inference operations over a Tensor.
    Batch {
        x: RawTensor<'a>,
        batch: Vec<(String, usize, usize)>,
    },
    /// A message to transmit tensors.
    Tensor(RawTensor<'a>),
}

impl<'a> Message<'a> {
    pub fn into_owned(self) -> Message<'static> {
        match self {
            Message::WorkerInfo(info) => Message::WorkerInfo(info.into_owned()),
            Message::Batch { x, batch } => Message::Batch {
                x: x.into_owned(),
                batch,
            },
            Message::Tensor(t) => Message::Tensor(t.into_owned()),
            Message::Hello => Message::Hello,
            Message::SingleOp {
                layer_name,
                x,
                index_pos,
                block_idx,
            } => Message::SingleOp {
                layer_name: Cow::Owned(layer_name.into_owned()),
                x: x.into_owned(),
                index_pos,
                block_idx,
            },
        }
    }
}

#[inline]
async fn read_u32be<R>(reader: &mut R) -> Result<u32>
where
    R: AsyncReadExt + Unpin,
{
    Ok(u32::from_be(reader.read_u32().await?))
}

impl Message<'_> {
    /// Create a Message::SingleOp message.
    pub fn single_op<'a, S>(
        layer_name: S,
        x: &'a Tensor,
        index_pos: usize,
        block_idx: usize,
    ) -> candle_core::Result<Message<'a>>
    where
        S: Into<Cow<'a, str>>,
    {
        Ok(Message::SingleOp {
            layer_name: layer_name.into(),
            x: x.try_into()?,
            index_pos,
            block_idx,
        })
    }

    /// Create a Message::Tensor message.
    pub fn from_tensor(x: &Tensor) -> candle_core::Result<Message<'static>> {
        x.try_into().map(Message::Tensor)
    }

    /// Create a Message::Batch message.
    pub fn from_batch(
        x: &Tensor,
        batch: Vec<(String, usize, usize)>,
    ) -> candle_core::Result<Message<'static>> {
        Ok(Message::Batch {
            x: x.try_into()?,
            batch,
        })
    }

    /// Deserializes a Message from raw bytes.
    fn from_slice(raw: &[u8]) -> Result<Message> {
        Ok(Message::read_from_buffer_with_ctx(
            BigEndian::default(),
            raw,
        )?)
    }

    fn from_bytes<B: AsRef<[u8]>>(raw: B) -> Result<Message<'static>> {
        Ok(Self::from_slice(raw.as_ref())?.into_owned())
    }

    // Yes, I could use GRPC, but this is simpler and faster.
    // Check speedy benchmarks ;)

    fn write_to_vec(&self, buf: &mut Vec<u8>) -> Result<()> {
        buf.extend_from_slice(&super::PROTO_MAGIC.to_be_bytes());

        let len_pos = buf.len();
        buf.extend_from_slice(&[0, 0, 0, 0]);

        let zero = buf.len();
        self.write_to_buffer_with_ctx(BigEndian::default(), buf)?;

        let len = buf.len() - zero;
        let Some(len): Option<u32> = len
            .try_into()
            .ok()
            .filter(|n| *n <= super::MESSAGE_MAX_SIZE)
        else {
            bail!("request {len} > MESSAGE_MAX_SIZE");
        };
        unsafe { *buf[len_pos..][..4].as_mut_ptr().cast::<[u8; 4]>() = len.to_be_bytes() };

        Ok(())
    }

    /// Serializes the message to raw bytes.
    fn to_vec(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.write_to_vec(&mut buf)?;
        Ok(buf)
    }

    /// Write a Message with the provided writer.
    pub async fn to_writer<W>(&self, writer: &mut W) -> Result<usize>
    where
        W: AsyncWriteExt + Unpin,
    {
        let req = self.to_vec()?;
        writer.write_all(&req).await?;
        Ok(req.len())
    }

    /// Read a Message with the provided reader.
    pub async fn from_reader<R>(reader: &mut R) -> Result<(usize, Message<'static>)>
    where
        R: AsyncReadExt + Unpin,
    {
        let magic = read_u32be(reader).await?;
        if magic != super::PROTO_MAGIC {
            bail!("invalid magic value: {magic}");
        }

        let req_size = read_u32be(reader).await?;
        if req_size > super::MESSAGE_MAX_SIZE {
            bail!("request size {req_size} > MESSAGE_MAX_SIZE");
        }

        let len: usize = req_size.try_into()?;
        let mut req = Vec::with_capacity(len);
        unsafe {
            reader
                .read_exact(core::slice::from_raw_parts_mut(req.as_mut_ptr(), len))
                .await?;
            req.set_len(len);
        }
        Ok((len, Message::from_bytes(req)?))
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, time::SystemTime};

    use candle_core::{DType, Device, Tensor};
    use speedy::{BigEndian, Readable, Writable};

    use super::RawTensor;

    macro_rules! ok {
        ($exp:expr $(,)?) => {
            match ($exp) {
                Ok(res) => res,
                Err(err) => panic!("{}: {}", err, stringify!($exp)),
            }
        };
    }

    fn tensor_eq(a: &Tensor, b: &Tensor) -> bool {
        fn rank1(t: &Tensor) -> Cow<Tensor> {
            let t = if t.rank() == 1 {
                Cow::Borrowed(t)
            } else {
                Cow::Owned(ok!(t.flatten_all()))
            };
            assert_eq!(t.rank(), 1);
            t
        }

        macro_rules! eql {
            ($a:expr, $b:expr, $meth:ident $(,)?) => {{
                let a = $a;
                let b = $b;
                if a.dims() != b.dims() {
                    false
                } else {
                    match a.dtype() {
                        DType::U8 => ok!(a.$meth::<u8>()) == ok!(b.$meth::<u8>()),
                        DType::U32 => ok!(a.$meth::<u32>()) == ok!(b.$meth::<u32>()),
                        DType::I64 => ok!(a.$meth::<i64>()) == ok!(b.$meth::<i64>()),
                        DType::BF16 => ok!(a.$meth::<half::bf16>()) == ok!(b.$meth::<half::bf16>()),
                        DType::F16 => ok!(a.$meth::<half::f16>()) == ok!(b.$meth::<half::f16>()),
                        DType::F32 => ok!(a.$meth::<f32>()) == ok!(b.$meth::<f32>()),
                        DType::F64 => ok!(a.$meth::<f64>()) == ok!(b.$meth::<f64>()),
                    }
                }
            }};
        }

        if a.dtype() != b.dtype() {
            return false;
        }

        if a.rank() == b.rank() {
            match a.rank() {
                0 => return eql!(a, b, to_vec0),
                2 => return eql!(a, b, to_vec2),
                3 => return eql!(a, b, to_vec3),
                _ => (),
            }
        }

        eql!(rank1(a), rank1(b), to_vec1)
    }

    fn rand_tensor(device: &Device) -> Tensor {
        let up = half::f16::from_f32((420 + 69) as f32);
        let lo = -up;
        if ok!(SystemTime::UNIX_EPOCH.elapsed()).as_millis() % 2 == 0 {
            ok!(Tensor::rand(lo, up, (2, 3), device))
        } else {
            ok!(Tensor::rand(lo, up, (2, 2, 3), device))
        }
    }

    macro_rules! assert_tensor_eq {
        ($a:expr, $b:expr $(,)?) => {{
            let a = &($a);
            let b = &($b);
            if !tensor_eq(a, b) {
                panic!("assertion `left == right` failed\n  left: {a:?}\n right: {b:?}");
            }
        }};
    }

    #[test]
    fn test_tensor_serde() {
        let device = Device::Cpu;
        let t1 = rand_tensor(&device);
        let rt1 = ok!(RawTensor::try_from(&t1));
        let rt2 = ok!(RawTensor::read_from_buffer_with_ctx(
            BigEndian::default(),
            &ok!(rt1.write_to_vec_with_ctx(BigEndian::default()))
        ))
        .into_owned();
        let t2 = ok!(rt2.clone().into_tensor(&device));
        assert_eq!(rt1, rt2);
        assert_tensor_eq!(t1, t2);
    }
}
