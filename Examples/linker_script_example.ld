/* Memories definition */
MEMORY
{
  ROM    (xrw)    : ORIGIN = 0x34180400,   LENGTH = 255K
  RAM    (xrw)    : ORIGIN = 0x341C0000,   LENGTH = 256K
  RAM_DTCM (xrw)  : ORIGIN = 0x20000000,   LENGTH = 128K
  
  RAM_BIG (xrw)   : ORIGIN = 0x24000000, LENGTH = 1024K
}

/* Sections */
SECTIONS
{ 
  .dtcm_data : 
  {	
  	  . = ALIGN(4);
      *(.dtcm_data)
      . = ALIGN(4);
  } >RAM_DTCM
  
  .big_data :
  {
	  . = ALIGN(32);
	  __big_data_start__ = .;
	  *(.big_data)
	  *(.big_data.*)
	  . = ALIGN(32);
	  __big_data_end__ = .;
  } > RAM_BIG
}
