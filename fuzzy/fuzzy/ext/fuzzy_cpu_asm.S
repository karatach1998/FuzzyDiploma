#define T_NUM 11

        .data
.LCPI0_0:
        .long   1065353216              # float 1
        .long   1065353216              # float 1
        .long   1065353216              # float 1
        .long   1065353216              # float 1
.LCPI0_2:
        .long   1063675494              # float 0.899999976
        .long   1063675494              # float 0.899999976
        .long   1063675494              # float 0.899999976
        .long   1063675494              # float 0.899999976
.LCPI0_3:
        .long   1061997773              # float 0.800000011
        .long   1061997773              # float 0.800000011
        .long   1061997773              # float 0.800000011
        .long   1061997773              # float 0.800000011
.LCPI0_4:
        .long   1060320051              # float 0.699999988
        .long   1060320051              # float 0.699999988
        .long   1060320051              # float 0.699999988
        .long   1060320051              # float 0.699999988
.LCPI0_5:
        .long   1058642330              # float 0.600000024
        .long   1058642330              # float 0.600000024
        .long   1058642330              # float 0.600000024
        .long   1058642330              # float 0.600000024
.LCPI0_6:
        .long   1056964608              # float 0.5
        .long   1056964608              # float 0.5
        .long   1056964608              # float 0.5
        .long   1056964608              # float 0.5
.LCPI0_7:
        .long   1053609164              # float 0.399999976
        .long   1053609164              # float 0.399999976
        .long   1053609164              # float 0.399999976
        .long   1053609164              # float 0.399999976
.LCPI0_8:
        .long   1050253722              # float 0.300000012
        .long   1050253722              # float 0.300000012
        .long   1050253722              # float 0.300000012
        .long   1050253722              # float 0.300000012
.LCPI0_9:
        .long   1045220556              # float 0.199999988
        .long   1045220556              # float 0.199999988
        .long   1045220556              # float 0.199999988
        .long   1045220556              # float 0.199999988
.LCPI0_10:
        .long   1036831952              # float 0.100000024
        .long   1036831952              # float 0.100000024
        .long   1036831952              # float 0.100000024
        .long   1036831952              # float 0.100000024
.LCPI0_21:
        .zero   16
.LCPI0_1:
        .long   1092616192              # float 10
.LCPI0_11:
        .long   1065353216              # float 1
.LCPI0_12:
        .long   1063675494              # float 0.899999976
.LCPI0_13:
        .long   1061997773              # float 0.800000011
.LCPI0_14:
        .long   1060320051              # float 0.699999988
.LCPI0_15:
        .long   1058642330              # float 0.600000024
.LCPI0_16:
        .long   1056964608              # float 0.5
.LCPI0_17:
        .long   1053609164              # float 0.399999976
.LCPI0_18:
        .long   1050253722              # float 0.300000012
.LCPI0_19:
        .long   1045220556              # float 0.199999988
.LCPI0_20:
        .long   1036831952              # float 0.100000024

        .text
        .globl predict_cpu_asm_clang
        .type predict_cpu_asm_clang, @function
predict_cpu_asm_clang:                  # @predict_cpu
        pushq   %rbp
        movq    %rsp, %rbp
        pushq   %r15
        pushq   %r14
        pushq   %r13
        pushq   %r12
        pushq   %rbx
        subq    $456, %rsp              # imm = 0x1C8
        movq    %r9, -160(%rbp)         # 8-byte Spill
        movq    %rcx, %r11
        movq    %rdi, -64(%rbp)         # 8-byte Spill
        movq    %rsp, -120(%rbp)        # 8-byte Spill
        movl    32(%rbp), %esi
        movl    24(%rbp), %r9d
        movq    16(%rbp), %rax
        movl    (%rdx,%rsi,4), %ecx
        movq    %rcx, -56(%rbp)         # 8-byte Spill
        leaq    15(,%rcx,4), %rcx
        andq    $-16, %rcx
        movq    %rsp, %r14
        subq    %rcx, %r14
        movq    %r14, %rsp
        movl    (%rdx,%rsi,4), %r12d
        testq   %r12, %r12
        je      .LBB0_12
        cmpl    $7, %r12d
        ja      .LBB0_4
        xorl    %ecx, %ecx
        jmp     .LBB0_3
.LBB0_4:
        movl    %r12d, %ecx
        andl    $-8, %ecx
        leaq    -8(%rcx), %rdi
        movq    %rdi, %rsi
        shrq    $3, %rsi
        addq    $1, %rsi
        movl    %esi, %ebx
        andl    $7, %ebx
        cmpq    $56, %rdi
        jae     .LBB0_6
        xorl    %esi, %esi
        testq   %rbx, %rbx
        jne     .LBB0_9
        jmp     .LBB0_11
.LBB0_6:
        movl    $1, %edi
        subq    %rsi, %rdi
        addq    %rbx, %rdi
        addq    $-1, %rdi
        xorl    %esi, %esi
        movaps  .LCPI0_0(%rip), %xmm0   # xmm0 = [1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00]
.LBB0_7:                                # =>This Inner Loop Header: Depth=1
        movups  %xmm0, (%rax,%rsi,4)
        movups  %xmm0, 16(%rax,%rsi,4)
        movups  %xmm0, 32(%rax,%rsi,4)
        movups  %xmm0, 48(%rax,%rsi,4)
        movups  %xmm0, 64(%rax,%rsi,4)
        movups  %xmm0, 80(%rax,%rsi,4)
        movups  %xmm0, 96(%rax,%rsi,4)
        movups  %xmm0, 112(%rax,%rsi,4)
        movups  %xmm0, 128(%rax,%rsi,4)
        movups  %xmm0, 144(%rax,%rsi,4)
        movups  %xmm0, 160(%rax,%rsi,4)
        movups  %xmm0, 176(%rax,%rsi,4)
        movups  %xmm0, 192(%rax,%rsi,4)
        movups  %xmm0, 208(%rax,%rsi,4)
        movups  %xmm0, 224(%rax,%rsi,4)
        movups  %xmm0, 240(%rax,%rsi,4)
        addq    $64, %rsi
        addq    $8, %rdi
        jne     .LBB0_7
        testq   %rbx, %rbx
        je      .LBB0_11
.LBB0_9:
        leaq    (%rax,%rsi,4), %rsi
        addq    $16, %rsi
        negq    %rbx
        movaps  .LCPI0_0(%rip), %xmm0   # xmm0 = [1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00]
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
        movups  %xmm0, -16(%rsi)
        movups  %xmm0, (%rsi)
        addq    $32, %rsi
        addq    $1, %rbx
        jne     .LBB0_10
.LBB0_11:
        cmpq    %r12, %rcx
        je      .LBB0_12
.LBB0_3:                                # =>This Inner Loop Header: Depth=1
        movl    $1065353216, (%rax,%rcx,4) # imm = 0x3F800000
        addq    $1, %rcx
        cmpq    %r12, %rcx
        jb      .LBB0_3
.LBB0_12:
        testl   %r9d, %r9d
        je      .LBB0_39
        shlq    $2, -56(%rbp)           # 8-byte Folded Spill
        movl    %r9d, %eax
        movq    %rax, -168(%rbp)        # 8-byte Spill
        movl    %r12d, %eax
        andl    $-8, %eax
        movq    %rax, -152(%rbp)        # 8-byte Spill
        leaq    -8(%rax), %rax
        movq    %rax, -144(%rbp)        # 8-byte Spill
        shrq    $3, %rax
        leal    1(%rax), %ecx
        movl    %r12d, %r13d
        andl    $-4, %r13d
        andl    $1, %ecx
        movq    %rcx, -136(%rbp)        # 8-byte Spill
        subq    %rax, %rcx
        addq    $-1, %rcx
        movq    %rcx, -128(%rbp)        # 8-byte Spill
        xorl    %eax, %eax
        movq    %rax, -48(%rbp)         # 8-byte Spill
        movq    %rdx, -176(%rbp)        # 8-byte Spill
        jmp     .LBB0_14
.LBB0_31:                               #   in Loop: Header=BB0_14 Depth=1
        xorl    %eax, %eax
        cmpq    $0, -136(%rbp)          # 8-byte Folded Reload
        jne     .LBB0_35
        jmp     .LBB0_36
.LBB0_14:                               # =>This Loop Header: Depth=1
        xorl    %esi, %esi
        movq    %r14, %rdi
        movq    -56(%rbp), %rdx         # 8-byte Reload
        movq    %r8, %rbx
        movq    %r11, %r15
        callq   memset@PLT
        movss   .LCPI0_17(%rip), %xmm15 # xmm15 = mem[0],zero,zero,zero
        movss   .LCPI0_16(%rip), %xmm14 # xmm14 = mem[0],zero,zero,zero
        movss   .LCPI0_15(%rip), %xmm13 # xmm13 = mem[0],zero,zero,zero
        movss   .LCPI0_14(%rip), %xmm12 # xmm12 = mem[0],zero,zero,zero
        movss   .LCPI0_13(%rip), %xmm2  # xmm2 = mem[0],zero,zero,zero
        xorps   %xmm7, %xmm7
        movss   .LCPI0_11(%rip), %xmm11 # xmm11 = mem[0],zero,zero,zero
        movq    %r15, %r11
        movq    -176(%rbp), %rdx        # 8-byte Reload
        movq    %rbx, %r8
        movl    32(%rbp), %eax
        testl   %eax, %eax
        je      .LBB0_27
        movq    -64(%rbp), %rax         # 8-byte Reload
        movl    32(%rbp), %ecx
        movq    (%rax,%rcx,8), %rax
        movq    -160(%rbp), %rcx        # 8-byte Reload
        movq    -48(%rbp), %rsi         # 8-byte Reload
        movzbl  (%rcx,%rsi), %ecx
        movq    (%rax,%rcx,8), %r9
        xorl    %r10d, %r10d
.LBB0_16:                               #   Parent Loop BB0_14 Depth=1
        movaps  %xmm2, %xmm8
        movq    -64(%rbp), %rax         # 8-byte Reload
        movq    (%rax,%r10,8), %rax
        movq    (%r8,%r10,8), %rcx
        movq    -48(%rbp), %rsi         # 8-byte Reload
        movzbl  (%rcx,%rsi), %ecx
        movq    (%rax,%rcx,8), %r15
        movq    (%r11,%r10,8), %rsi
        xorps   %xmm0, %xmm0
        movups  %xmm0, -84(%rbp)
        movaps  %xmm0, -96(%rbp)
        movaps  %xmm0, -112(%rbp)
        movl    (%rdx,%r10,4), %edi
        cmpl    $1, %edi
        movss   .LCPI0_1(%rip), %xmm9   # xmm9 = mem[0],zero,zero,zero
        je      .LBB0_24
        addl    $-1, %edi
        movss   (%r15), %xmm0           # xmm0 = mem[0],zero,zero,zero
        xorl    %ebx, %ebx
.LBB0_18:                               #   Parent Loop BB0_14 Depth=1
        movl    %ebx, %eax
        xorps   %xmm1, %xmm1
        cvtsi2ssq       %rax, %xmm1
        divss   %xmm9, %xmm1
        movaps  %xmm0, %xmm2
        xorl    %eax, %eax
.LBB0_19:                               #   Parent Loop BB0_14 Depth=1
        movq    %rax, %rcx
        movaps  %xmm2, %xmm4
        leaq    1(%rax), %rax
        movss   4(%r15,%rcx,4), %xmm2   # xmm2 = mem[0],zero,zero,zero
        movaps  %xmm1, %xmm3
        subss   %xmm4, %xmm3
        movaps  %xmm2, %xmm5
        subss   %xmm1, %xmm5
        mulss   %xmm3, %xmm5
        ucomiss %xmm7, %xmm5
        jb      .LBB0_22
        movss   (%rsi,%rcx,4), %xmm5    # xmm5 = mem[0],zero,zero,zero
        movss   4(%rsi,%rcx,4), %xmm6   # xmm6 = mem[0],zero,zero,zero
        subss   %xmm5, %xmm6
        mulss   %xmm6, %xmm3
        movaps  %xmm2, %xmm6
        subss   %xmm4, %xmm6
        divss   %xmm6, %xmm3
        addss   %xmm5, %xmm3
        ucomiss -112(%rbp,%rbx,4), %xmm3
        jbe     .LBB0_22
        movss   %xmm3, -112(%rbp,%rbx,4)
.LBB0_22:                               #   in Loop: Header=BB0_19 Depth=4
        cmpq    %rdi, %rax
        jb      .LBB0_19
        addq    $1, %rbx
        cmpq    $11, %rbx
        jne     .LBB0_18
.LBB0_24:                               #   in Loop: Header=BB0_16 Depth=2
        testl   %r12d, %r12d
        movaps  %xmm11, %xmm1
        movaps  %xmm8, %xmm2
        je      .LBB0_44
        movaps  %xmm1, %xmm8
        movss   -112(%rbp), %xmm2       # xmm2 = mem[0],zero,zero,zero
        movss   -108(%rbp), %xmm0       # xmm0 = mem[0],zero,zero,zero
        movss   -104(%rbp), %xmm1       # xmm1 = mem[0],zero,zero,zero
        movss   -100(%rbp), %xmm3       # xmm3 = mem[0],zero,zero,zero
        movss   -96(%rbp), %xmm12       # xmm12 = mem[0],zero,zero,zero
        movss   -92(%rbp), %xmm5        # xmm5 = mem[0],zero,zero,zero
        movss   -88(%rbp), %xmm11       # xmm11 = mem[0],zero,zero,zero
        movss   -84(%rbp), %xmm6        # xmm6 = mem[0],zero,zero,zero
        movss   -80(%rbp), %xmm4        # xmm4 = mem[0],zero,zero,zero
        movss   -76(%rbp), %xmm9        # xmm9 = mem[0],zero,zero,zero
        movss   -72(%rbp), %xmm10       # xmm10 = mem[0],zero,zero,zero
        cmpl    $4, %r12d
        movaps  %xmm2, -496(%rbp)       # 16-byte Spill
        movaps  %xmm11, -480(%rbp)      # 16-byte Spill
        movaps  %xmm6, -464(%rbp)       # 16-byte Spill
        movaps  %xmm4, -448(%rbp)       # 16-byte Spill
        movaps  %xmm9, -432(%rbp)       # 16-byte Spill
        movaps  %xmm10, -416(%rbp)      # 16-byte Spill
        jae     .LBB0_40
        xorl    %eax, %eax
        movaps  %xmm5, %xmm9
        movaps  %xmm0, %xmm5
        movaps  %xmm1, %xmm10
        movaps  %xmm3, %xmm11
        movaps  %xmm12, %xmm6
        movaps  %xmm8, %xmm1
        movss   .LCPI0_13(%rip), %xmm2  # xmm2 = mem[0],zero,zero,zero
        movss   .LCPI0_14(%rip), %xmm12 # xmm12 = mem[0],zero,zero,zero
        jmp     .LBB0_43
.LBB0_40:                               #   in Loop: Header=BB0_16 Depth=2
        movaps  %xmm2, %xmm7
        shufps  $0, %xmm2, %xmm7        # xmm7 = xmm7[0,0],xmm2[0,0]
        movaps  %xmm7, -400(%rbp)       # 16-byte Spill
        movaps  %xmm0, -256(%rbp)       # 16-byte Spill
        shufps  $0, %xmm0, %xmm0        # xmm0 = xmm0[0,0,0,0]
        movaps  %xmm0, -384(%rbp)       # 16-byte Spill
        movaps  %xmm1, -240(%rbp)       # 16-byte Spill
        shufps  $0, %xmm1, %xmm1        # xmm1 = xmm1[0,0,0,0]
        movaps  %xmm1, -368(%rbp)       # 16-byte Spill
        movaps  %xmm3, -224(%rbp)       # 16-byte Spill
        shufps  $0, %xmm3, %xmm3        # xmm3 = xmm3[0,0,0,0]
        movaps  %xmm3, -352(%rbp)       # 16-byte Spill
        movaps  %xmm12, -208(%rbp)      # 16-byte Spill
        shufps  $0, %xmm12, %xmm12      # xmm12 = xmm12[0,0,0,0]
        movaps  %xmm12, -336(%rbp)      # 16-byte Spill
        movaps  %xmm5, -192(%rbp)       # 16-byte Spill
        shufps  $0, %xmm5, %xmm5        # xmm5 = xmm5[0,0,0,0]
        movaps  %xmm5, -320(%rbp)       # 16-byte Spill
        shufps  $0, %xmm11, %xmm11      # xmm11 = xmm11[0,0,0,0]
        movaps  %xmm11, -304(%rbp)      # 16-byte Spill
        shufps  $0, %xmm6, %xmm6        # xmm6 = xmm6[0,0,0,0]
        movaps  %xmm6, -288(%rbp)       # 16-byte Spill
        shufps  $0, %xmm4, %xmm4        # xmm4 = xmm4[0,0,0,0]
        movaps  %xmm4, -272(%rbp)       # 16-byte Spill
        movaps  %xmm9, %xmm13
        shufps  $0, %xmm9, %xmm13       # xmm13 = xmm13[0,0],xmm9[0,0]
        movaps  %xmm10, %xmm12
        shufps  $0, %xmm10, %xmm12      # xmm12 = xmm12[0,0],xmm10[0,0]
        xorl    %eax, %eax
        movaps  .LCPI0_0(%rip), %xmm2   # xmm2 = [1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00]
        movaps  .LCPI0_2(%rip), %xmm1   # xmm1 = [9.000000e-01,9.000000e-01,9.000000e-01,9.000000e-01]
        movaps  .LCPI0_3(%rip), %xmm4   # xmm4 = [8.000000e-01,8.000000e-01,8.000000e-01,8.000000e-01]
        movaps  .LCPI0_4(%rip), %xmm5   # xmm5 = [7.000000e-01,7.000000e-01,7.000000e-01,7.000000e-01]
        movaps  .LCPI0_5(%rip), %xmm6   # xmm6 = [6.000000e-01,6.000000e-01,6.000000e-01,6.000000e-01]
        movaps  .LCPI0_6(%rip), %xmm7   # xmm7 = [5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01]
        movaps  .LCPI0_7(%rip), %xmm8   # xmm8 = [4.000000e-01,4.000000e-01,4.000000e-01,4.000000e-01]
        movaps  .LCPI0_8(%rip), %xmm9   # xmm9 = [3.000000e-01,3.000000e-01,3.000000e-01,3.000000e-01]
        movaps  .LCPI0_9(%rip), %xmm10  # xmm10 = [2.000000e-01,2.000000e-01,2.000000e-01,2.000000e-01]
        movaps  .LCPI0_10(%rip), %xmm11 # xmm11 = [1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01]
.LBB0_41:                               #   Parent Loop BB0_14 Depth=1
        movups  (%r9,%rax,4), %xmm15
        movaps  %xmm15, %xmm0
        addps   %xmm2, %xmm0
        movaps  -400(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        xorps   %xmm14, %xmm14
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm1, %xmm0
        movaps  -384(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm4, %xmm0
        movaps  -368(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm5, %xmm0
        movaps  -352(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm6, %xmm0
        movaps  -336(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm7, %xmm0
        movaps  -320(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm8, %xmm0
        movaps  -304(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm9, %xmm0
        movaps  -288(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm10, %xmm0
        movaps  -272(%rbp), %xmm3       # 16-byte Reload
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        movaps  %xmm15, %xmm0
        addps   %xmm11, %xmm0
        movaps  %xmm13, %xmm3
        minps   %xmm0, %xmm3
        maxps   %xmm3, %xmm14
        addps   .LCPI0_21(%rip), %xmm15
        movaps  %xmm12, %xmm0
        minps   %xmm15, %xmm0
        maxps   %xmm0, %xmm14
        movaps  (%r14,%rax,4), %xmm0
        maxps   %xmm14, %xmm0
        movaps  %xmm0, (%r14,%rax,4)
        addq    $4, %rax
        cmpq    %rax, %r13
        jne     .LBB0_41
        movq    %r13, %rax
        cmpq    %r12, %r13
        movss   .LCPI0_11(%rip), %xmm8  # xmm8 = mem[0],zero,zero,zero
        movaps  %xmm8, %xmm1
        xorps   %xmm7, %xmm7
        movss   .LCPI0_13(%rip), %xmm2  # xmm2 = mem[0],zero,zero,zero
        movss   .LCPI0_14(%rip), %xmm12 # xmm12 = mem[0],zero,zero,zero
        movss   .LCPI0_15(%rip), %xmm13 # xmm13 = mem[0],zero,zero,zero
        movss   .LCPI0_16(%rip), %xmm14 # xmm14 = mem[0],zero,zero,zero
        movss   .LCPI0_17(%rip), %xmm15 # xmm15 = mem[0],zero,zero,zero
        movaps  -256(%rbp), %xmm5       # 16-byte Reload
        movaps  -240(%rbp), %xmm10      # 16-byte Reload
        movaps  -224(%rbp), %xmm11      # 16-byte Reload
        movaps  -208(%rbp), %xmm6       # 16-byte Reload
        movaps  -192(%rbp), %xmm9       # 16-byte Reload
        je      .LBB0_44
.LBB0_43:                               #   Parent Loop BB0_14 Depth=1
        movss   (%r9,%rax,4), %xmm4     # xmm4 = mem[0],zero,zero,zero
        movaps  %xmm4, %xmm0
        addss   %xmm1, %xmm0
        movaps  %xmm1, %xmm8
        movaps  -496(%rbp), %xmm1       # 16-byte Reload
        minss   %xmm0, %xmm1
        xorps   %xmm3, %xmm3
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   .LCPI0_12(%rip), %xmm0
        movaps  %xmm5, %xmm1
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   %xmm2, %xmm0
        movaps  %xmm10, %xmm1
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   %xmm12, %xmm0
        movaps  %xmm11, %xmm1
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   %xmm13, %xmm0
        movaps  %xmm6, %xmm1
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   %xmm14, %xmm0
        movaps  %xmm9, %xmm1
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   %xmm15, %xmm0
        movaps  -480(%rbp), %xmm1       # 16-byte Reload
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   .LCPI0_18(%rip), %xmm0
        movaps  -464(%rbp), %xmm1       # 16-byte Reload
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   .LCPI0_19(%rip), %xmm0
        movaps  -448(%rbp), %xmm1       # 16-byte Reload
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm4, %xmm0
        addss   .LCPI0_20(%rip), %xmm0
        movaps  -432(%rbp), %xmm1       # 16-byte Reload
        minss   %xmm0, %xmm1
        maxss   %xmm1, %xmm3
        movaps  %xmm8, %xmm1
        addss   %xmm7, %xmm4
        movaps  -416(%rbp), %xmm0       # 16-byte Reload
        minss   %xmm4, %xmm0
        maxss   %xmm0, %xmm3
        movss   (%r14,%rax,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
        maxss   %xmm3, %xmm0
        movss   %xmm0, (%r14,%rax,4)
        addq    $1, %rax
        cmpq    %rax, %r12
        jne     .LBB0_43
.LBB0_44:                               #   in Loop: Header=BB0_16 Depth=2
        addq    $1, %r10
        movl    32(%rbp), %eax
        cmpq    %rax, %r10
        movaps  %xmm1, %xmm11
        jne     .LBB0_16
.LBB0_27:                               #   in Loop: Header=BB0_14 Depth=1
        testl   %r12d, %r12d
        movq    16(%rbp), %rsi
        je      .LBB0_38
        cmpl    $7, %r12d
        ja      .LBB0_30
        xorl    %eax, %eax
        jmp     .LBB0_37
.LBB0_30:                               #   in Loop: Header=BB0_14 Depth=1
        cmpq    $0, -144(%rbp)          # 8-byte Folded Reload
        je      .LBB0_31
        movq    -128(%rbp), %rcx        # 8-byte Reload
        xorl    %eax, %eax
.LBB0_33:                               #   Parent Loop BB0_14 Depth=1
        movups  (%rsi,%rax,4), %xmm0
        movups  16(%rsi,%rax,4), %xmm1
        movups  32(%rsi,%rax,4), %xmm2
        movups  48(%rsi,%rax,4), %xmm3
        minps   (%r14,%rax,4), %xmm0
        minps   16(%r14,%rax,4), %xmm1
        movups  %xmm0, (%rsi,%rax,4)
        movups  %xmm1, 16(%rsi,%rax,4)
        minps   32(%r14,%rax,4), %xmm2
        minps   48(%r14,%rax,4), %xmm3
        movups  %xmm2, 32(%rsi,%rax,4)
        movups  %xmm3, 48(%rsi,%rax,4)
        addq    $16, %rax
        addq    $2, %rcx
        jne     .LBB0_33
        cmpq    $0, -136(%rbp)          # 8-byte Folded Reload
        je      .LBB0_36
.LBB0_35:                               #   in Loop: Header=BB0_14 Depth=1
        movups  (%rsi,%rax,4), %xmm0
        movups  16(%rsi,%rax,4), %xmm1
        minps   (%r14,%rax,4), %xmm0
        minps   16(%r14,%rax,4), %xmm1
        movups  %xmm0, (%rsi,%rax,4)
        movups  %xmm1, 16(%rsi,%rax,4)
.LBB0_36:                               #   in Loop: Header=BB0_14 Depth=1
        movq    -152(%rbp), %rcx        # 8-byte Reload
        movq    %rcx, %rax
        cmpq    %r12, %rcx
        je      .LBB0_38
.LBB0_37:                               #   Parent Loop BB0_14 Depth=1
        movss   (%rsi,%rax,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
        minss   (%r14,%rax,4), %xmm0
        movss   %xmm0, (%rsi,%rax,4)
        addq    $1, %rax
        cmpq    %rax, %r12
        jne     .LBB0_37
.LBB0_38:                               #   in Loop: Header=BB0_14 Depth=1
        movq    -48(%rbp), %rax         # 8-byte Reload
        movq    %rax, %rcx
        addq    $1, %rcx
        movq    %rcx, %rax
        movq    %rcx, -48(%rbp)         # 8-byte Spill
        cmpq    -168(%rbp), %rcx        # 8-byte Folded Reload
        jne     .LBB0_14
.LBB0_39:
        movq    -120(%rbp), %rsp        # 8-byte Reload
        leaq    -40(%rbp), %rsp
        popq    %rbx
        popq    %r12
        popq    %r13
        popq    %r14
        popq    %r15
        popq    %rbp
        retq
